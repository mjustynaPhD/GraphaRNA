"""
Hybrid Training Loop — Supervised + Reinforcement Learning.

Phase 1 (Supervised):
    Train KinematicGNN + LDDTSuggester to mimic known structures.
    Loss = Frame alignment loss + pLDDT prediction loss.

Phase 2 (RL Fine-tuning):
    Use PPO to minimize physical violations.
    Reward = R = w_bp·BasePairs - w_clash·Clashes - w_torsion·TorsionStress.

Can also run both phases jointly with a curriculum schedule.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from typing import Optional, Dict, Tuple

from grapharna.kinematic.frames import (
    RigidFrame,
    compute_local_frames_from_coords,
    initialize_linear_chain,
)
from grapharna.kinematic.curriculum_scheduler import (
    TrainingScheduler,
)
from grapharna.kinematic.kinematic_gnn import KinematicGNN, KinematicConfig
from grapharna.kinematic.lddt_suggester import LDDTSuggester
from grapharna.kinematic.recursive_refiner import RecursiveRefiner
from grapharna.kinematic.refinement_loop import (
    RecursiveRefinementLoop,
    RefinementReward,
)
from grapharna.kinematic.rl_agent import (
    PPOAgent,
    RNARefinementEnv,
    RewardFunction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supervised Losses
# ---------------------------------------------------------------------------

def frame_alignment_loss(
    frames_pred: RigidFrame,
    coords_true: torch.Tensor,  # (N, 5, 3)
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute FAPE-like loss between predicted frames and true coordinates.

    Frame Aligned Point Error (FAPE), adapted from AlphaFold2:
    For each pair of residues (i, j), compare the distance between
    atom positions in the predicted vs true structure, as measured
    in frame i's coordinate system.

    Simplified version: direct coordinate RMSD + rotation alignment.
    """
    coords_pred = frames_pred.global_coords()  # (N, 5, 3)

    if mask is not None:
        coords_pred = coords_pred[mask]
        coords_true = coords_true[mask]

    # L2 loss on atom positions
    coord_loss = F.smooth_l1_loss(coords_pred, coords_true)

    # Rotation alignment loss: compare frame orientations
    # Not applied here since we don't have ground-truth rotations
    # in every scenario — can be added when preprocessing stores frames

    return coord_loss


def supervised_loss(
    refiner: RecursiveRefiner,
    frames_init: RigidFrame,
    coords_true: torch.Tensor,
    sequences: list,
    residue_types: torch.Tensor,
    batch: torch.Tensor,
    covalent_edges: Optional[torch.Tensor] = None,
    bp_edges: Optional[torch.Tensor] = None,
    chain_edges: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute supervised training loss.

    Returns:
        dict with 'total_loss', 'coord_loss', 'plddt_loss', and 'outputs'.
    """
    # Run refinement
    outputs = refiner(
        frames=frames_init,
        sequences=sequences,
        residue_types=residue_types,
        batch=batch,
        covalent_edges=covalent_edges,
        bp_edges=bp_edges,
        chain_edges=chain_edges,
    )

    # Coordinate alignment loss
    coord_loss = frame_alignment_loss(outputs['frames'], coords_true, mask)

    # pLDDT prediction loss
    plddt_loss = refiner.lddt_suggester.compute_lddt_loss(
        pred_plddt=outputs['plddt'],
        coords_pred=outputs['coords'],
        coords_true=coords_true,
    )

    total_loss = coord_loss + 0.1 * plddt_loss

    return {
        'total_loss': total_loss,
        'coord_loss': coord_loss,
        'plddt_loss': plddt_loss,
        'outputs': outputs,
    }


# ---------------------------------------------------------------------------
# Data Conversion Utilities
# ---------------------------------------------------------------------------

def pyg_data_to_kinematic(
    data,
    sequence: str,
    device: torch.device,
) -> Tuple[RigidFrame, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert PyG Data object to kinematic refinement inputs.

    The existing GraphaRNA dataset stores atoms as a flat tensor with
    15 features per atom: [x, y, z, atom_type×4, res_type×4, c4', c2, c4/c6, n1/n9].

    This function:
    1. Extracts the 5 CG atoms per residue
    2. Reshapes into (N_residues, 5, 3) coordinates
    3. Computes local frames via Gram-Schmidt
    4. Extracts residue types and edge information

    Args:
        data:     PyG Data object from RNAPDBDataset.
        sequence: Nucleotide sequence string.
        device:   Target device.

    Returns:
        frames_true:    RigidFrame from ground-truth coordinates.
        residue_types:  (N_res,) int tensor.
        covalent_edges: (2, E_cov) backbone edges (residue-level).
        bp_edges:       (2, E_bp) base-pair edges (residue-level).
        chain_edges:    (2, N_res-1) sequential chain edges.
    """
    x = data.x.to(device)  # (N_atoms, 15)
    N_atoms = x.shape[0]
    N_res = N_atoms // 5  # 5 atoms per residue

    # Extract coordinates (first 3 features, scaled back to Å)
    coords_flat = x[:, :3] * 10.0  # Undo the /10 scaling from preprocessing
    coords = coords_flat.reshape(N_res, 5, 3)  # (N_res, 5, 3)

    # Extract residue types from one-hot (features 7:11)
    res_onehot = x[::5, 7:11]  # Take every 5th atom (one per residue)
    residue_types = res_onehot.argmax(dim=-1).long()  # (N_res,)

    # Compute frames from coordinates
    R, t, local_coords = compute_local_frames_from_coords(coords, residue_types)
    frames_true = RigidFrame(R=R, t=t, local_coords=local_coords)

    # Convert atom-level edges to residue-level edges
    edge_index = data.edge_index.to(device)  # (2, E)
    # Edges are at atom level; convert to residue level
    res_edges = edge_index // 5  # Map atom index → residue index
    # Remove intra-residue edges
    inter_mask = res_edges[0] != res_edges[1]
    res_edges = res_edges[:, inter_mask]
    # Remove duplicates
    res_edge_pairs = res_edges[0] * N_res + res_edges[1]
    unique_pairs, unique_idx = torch.unique(res_edge_pairs, return_inverse=True)
    first_occ = torch.zeros(unique_pairs.shape[0], dtype=torch.long, device=device)
    for i in range(res_edge_pairs.shape[0] - 1, -1, -1):
        first_occ[unique_idx[i]] = i
    res_edges = res_edges[:, first_occ]

    # Classify edges
    if data.edge_attr is not None:
        edge_attr = data.edge_attr.to(device)
        edge_attr_res = edge_attr[inter_mask][first_occ]

        # Edge type: [covalent, 2D-structure, spatial]
        # In GraphaRNA: type 0 = covalent, type 1 = base-pair, type 2 = spatial
        cov_mask = edge_attr_res[:, 0] > 0.5
        bp_mask = edge_attr_res[:, 1] > 0.5

        covalent_edges = res_edges[:, cov_mask]
        bp_edges = res_edges[:, bp_mask]
    else:
        # Default: sequential edges as covalent
        covalent_edges = torch.stack([
            torch.arange(N_res - 1, device=device),
            torch.arange(1, N_res, device=device),
        ])
        bp_edges = torch.zeros(2, 0, dtype=torch.long, device=device)

    # Sequential chain edges (always present)
    chain_edges = torch.stack([
        torch.arange(N_res - 1, device=device),
        torch.arange(1, N_res, device=device),
    ])

    return frames_true, residue_types, covalent_edges, bp_edges, chain_edges


# ---------------------------------------------------------------------------
# Hybrid Trainer
# ---------------------------------------------------------------------------

class HybridTrainer:
    """Manages the two-phase training loop.

    Phase 1: Supervised pre-training
        - Initialize from linear chain → refine → compare to true structure
        - Train KinematicGNN + LDDTSuggester jointly

    Phase 2: RL fine-tuning
        - Freeze LDDTSuggester
        - Use PPO with physics-based reward
        - Focus on reducing clashes and improving base-pair geometry
    """

    def __init__(
        self,
        config: KinematicConfig,
        supervised_epochs: int = 100,
        rl_epochs: int = 50,
        supervised_lr: float = 1e-3,
        rl_lr: float = 3e-4,
        batch_size: int = 4,
        save_dir: str = './save',
        wandb_project: Optional[str] = None,
        device: torch.device = None,
    ):
        self.config = config
        self.supervised_epochs = supervised_epochs
        self.rl_epochs = rl_epochs
        self.supervised_lr = supervised_lr
        self.rl_lr = rl_lr
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Build model components
        self.gnn = KinematicGNN(config).to(self.device)
        self.lddt_suggester = LDDTSuggester(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.reward_fn = RefinementReward().to(self.device)

        # Build recursive refinement loop (new architecture)
        self.refiner = RecursiveRefinementLoop(
            gnn=self.gnn,
            lddt_suggester=self.lddt_suggester,
            reward_fn=self.reward_fn,
            max_iterations=config.max_refinement_steps,
            lever_damping=config.lever_damping,
        ).to(self.device)

        # Keep legacy RecursiveRefiner only for backward-compat loading
        self._legacy_refiner: Optional[RecursiveRefiner] = None

        # Curriculum noise scheduler
        self.training_scheduler = TrainingScheduler(
            sigma_init=0.05,
            sigma_min=0.01,
            sigma_max=1.5,
            sigma_growth_factor=1.3,
            bp_threshold=0.80,
            translation_ratio=3.0,
            warmup_steps=20,
            reward_window=50,
        )

        # RL agent (initialized later)
        self.rl_agent: Optional[PPOAgent] = None
        self.rl_env: Optional[RNARefinementEnv] = None

    def train_supervised(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        exp_name: str = 'kinematic_supervised',
    ) -> Dict[str, list]:
        """Phase 1: Supervised pre-training.

        Returns:
            Training history dict.
        """
        logger.info("=" * 60)
        logger.info("Phase 1: Supervised Pre-training")
        logger.info("=" * 60)

        optimizer = optim.Adam(
            list(self.gnn.parameters()) + list(self.lddt_suggester.parameters()),
            lr=self.supervised_lr,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.supervised_epochs, eta_min=1e-6
        )

        history = {
            'train_loss': [], 'train_coord_loss': [], 'train_plddt_loss': [],
            'val_loss': [],
        }

        for epoch in range(self.supervised_epochs):
            self.refiner.train()
            epoch_losses = []
            epoch_coord_losses = []
            epoch_plddt_losses = []

            for data, name, seqs in train_loader:
                optimizer.zero_grad()

                # Convert to kinematic format
                frames_true, residue_types, cov_edges, bp_edges, chain_edges = (
                    pyg_data_to_kinematic(data, seqs, self.device)
                )

                batch = data.batch.to(self.device)
                # Map atom-level batch to residue-level batch
                res_batch = batch[::5]

                # Initialize from noisy version of true structure
                # (curriculum: σ starts low and increases adaptively)
                frames_init = self.training_scheduler.apply_noise(frames_true)

                # Compute supervised loss
                if isinstance(seqs, str):
                    seqs = [seqs]
                loss_dict = supervised_loss(
                    refiner=self.refiner,
                    frames_init=frames_init,
                    coords_true=frames_true.global_coords(),
                    sequences=seqs,
                    residue_types=residue_types,
                    batch=res_batch,
                    covalent_edges=cov_edges,
                    bp_edges=bp_edges,
                    chain_edges=chain_edges,
                )

                loss = loss_dict['total_loss']
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.gnn.parameters()) + list(self.lddt_suggester.parameters()),
                    2.0
                )
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_coord_losses.append(loss_dict['coord_loss'].item())
                epoch_plddt_losses.append(loss_dict['plddt_loss'].item())

            scheduler.step()

            # Log
            mean_loss = np.mean(epoch_losses)
            history['train_loss'].append(mean_loss)
            history['train_coord_loss'].append(np.mean(epoch_coord_losses))
            history['train_plddt_loss'].append(np.mean(epoch_plddt_losses))

            # Estimate BP recovery for curriculum (use coord loss as proxy:
            # lower coord loss → better structure → higher BP recovery).
            # A proper BP recovery requires running the reward function,
            # which we do every ``curriculum_eval_every`` epochs.
            if (epoch + 1) % max(1, self.supervised_epochs // 20) == 0:
                bp_recovery = self._estimate_bp_recovery(train_loader)
                self.training_scheduler.log_step(
                    reward=0.0,  # no RL reward in supervised phase
                    bp_recovery=bp_recovery,
                )
                history.setdefault('bp_recovery', []).append(bp_recovery)
                history.setdefault('curriculum_sigma', []).append(
                    self.training_scheduler.current_sigma
                )

            # Validation
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                logger.info(
                    f"Epoch {epoch+1}/{self.supervised_epochs} | "
                    f"Train Loss: {mean_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"σ: {self.training_scheduler.current_sigma:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.supervised_epochs} | "
                    f"Train Loss: {mean_loss:.4f} | "
                    f"σ: {self.training_scheduler.current_sigma:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1, exp_name, phase='supervised')

        return history

    def train_rl(
        self,
        train_loader: DataLoader,
        exp_name: str = 'kinematic_rl',
    ) -> Dict[str, list]:
        """Phase 2: RL fine-tuning with PPO.

        Returns:
            Training history dict.
        """
        logger.info("=" * 60)
        logger.info("Phase 2: RL Fine-tuning (PPO)")
        logger.info("=" * 60)

        # Initialize RL components
        self.rl_env = RNARefinementEnv(
            reward_fn=self.reward_fn,
            max_steps=self.config.max_refinement_steps,
            lever_damping=self.config.lever_damping,
        )

        self.rl_agent = PPOAgent(
            policy=self.gnn,
            config=self.config,
            lr=self.rl_lr,
        )
        self.rl_agent.to(self.device)

        # Freeze LDDTSuggester during RL
        for param in self.lddt_suggester.parameters():
            param.requires_grad = False

        history = {
            'episode_reward': [], 'bp_reward': [],
            'clash_penalty': [], 'torsion_penalty': [],
            'policy_loss': [], 'value_loss': [],
        }

        for epoch in range(self.rl_epochs):
            epoch_rewards = []
            epoch_bp = []
            epoch_clash = []
            epoch_torsion = []

            for data, name, seqs in train_loader:
                # Convert data
                frames_true, residue_types, cov_edges, bp_edges, chain_edges = (
                    pyg_data_to_kinematic(data, seqs, self.device)
                )
                batch = data.batch.to(self.device)
                res_batch = batch[::5]

                # Initialize from noisy ground truth (curriculum) or
                # linear chain at max difficulty
                if isinstance(seqs, str):
                    seqs = [seqs]

                if self.training_scheduler.difficulty_progress < 1.0:
                    # Curriculum: start from noisy version of true structure
                    frames_init = self.training_scheduler.apply_noise(frames_true)
                else:
                    # Max difficulty: fold from scratch
                    frames_init = initialize_linear_chain(
                        "".join(seqs), device=self.device
                    )

                # Reset environment
                self.rl_env.reset(
                    frames=frames_init,
                    bp_edges=bp_edges,
                    chain_edges=chain_edges,
                    batch=res_batch,
                )

                # Collect rollout
                stats = self.rl_agent.collect_rollout(
                    env=self.rl_env,
                    sequences=seqs,
                    residue_types=residue_types,
                    batch=res_batch,
                    covalent_edges=cov_edges,
                    bp_edges=bp_edges,
                    chain_edges=chain_edges,
                )

                epoch_rewards.append(stats['total_reward'])
                epoch_bp.append(stats['final_bp_reward'])
                epoch_clash.append(stats['final_clash_penalty'])
                epoch_torsion.append(stats['final_torsion_penalty'])

            # PPO update
            update_stats = self.rl_agent.update()

            # --- Curriculum: log step with BP recovery → may promote σ ---
            mean_bp = float(np.mean(epoch_bp))
            self.training_scheduler.log_step(
                reward=float(np.mean(epoch_rewards)),
                bp_recovery=mean_bp,
            )

            # Log
            history['episode_reward'].append(np.mean(epoch_rewards))
            history['bp_reward'].append(np.mean(epoch_bp))
            history['clash_penalty'].append(np.mean(epoch_clash))
            history['torsion_penalty'].append(np.mean(epoch_torsion))
            history['policy_loss'].append(update_stats.get('policy_loss', 0))
            history['value_loss'].append(update_stats.get('value_loss', 0))
            history.setdefault('curriculum_sigma', []).append(
                self.training_scheduler.current_sigma
            )

            curriculum_stats = self.training_scheduler.get_stats()
            logger.info(
                f"RL Epoch {epoch+1}/{self.rl_epochs} | "
                f"Reward: {np.mean(epoch_rewards):.4f} | "
                f"BP: {mean_bp:.4f} | "
                f"Clash: {np.mean(epoch_clash):.4f} | "
                f"Torsion: {np.mean(epoch_torsion):.4f} | "
                f"σ: {curriculum_stats['curriculum/sigma']:.4f} | "
                f"Difficulty: {curriculum_stats['curriculum/difficulty_progress']:.1%}"
            )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1, exp_name, phase='rl')

        return history

    def train_hybrid(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        exp_name: str = 'kinematic_hybrid',
    ) -> Dict[str, Dict]:
        """Run full hybrid training: supervised → RL.

        Returns:
            dict with 'supervised' and 'rl' history dicts.
        """
        # Phase 1: Supervised
        sup_history = self.train_supervised(
            train_loader, val_loader, exp_name + '_supervised'
        )

        # Phase 2: RL
        rl_history = self.train_rl(
            train_loader, exp_name + '_rl'
        )

        return {
            'supervised': sup_history,
            'rl': rl_history,
        }

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return mean loss."""
        self.refiner.eval()
        losses = []

        for data, name, seqs in val_loader:
            frames_true, residue_types, cov_edges, bp_edges, chain_edges = (
                pyg_data_to_kinematic(data, seqs, self.device)
            )
            batch = data.batch.to(self.device)
            res_batch = batch[::5]

            frames_init = self._noise_frames(frames_true, noise_scale=0.5)

            if isinstance(seqs, str):
                seqs = [seqs]
            loss_dict = supervised_loss(
                refiner=self.refiner,
                frames_init=frames_init,
                coords_true=frames_true.global_coords(),
                sequences=seqs,
                residue_types=residue_types,
                batch=res_batch,
                covalent_edges=cov_edges,
                bp_edges=bp_edges,
                chain_edges=chain_edges,
            )
            losses.append(loss_dict['total_loss'].item())

        self.refiner.train()
        return np.mean(losses)

    def _noise_frames(
        self,
        frames: RigidFrame,
        noise_scale: Optional[float] = None,
    ) -> RigidFrame:
        """Add noise to frames for training initialization.

        Delegates to the CurriculumNoiseScheduler for proper SO(3)
        Gaussian noise injection.  If ``noise_scale`` is given it
        overrides the scheduler's current σ.
        """
        return self.training_scheduler.apply_noise(frames, sigma_override=noise_scale)

    @torch.no_grad()
    def _estimate_bp_recovery(
        self,
        loader: DataLoader,
        max_samples: int = 8,
    ) -> float:
        """Estimate BP recovery rate on a few training samples.

        Runs the current model on noisy structures and measures how
        many native base-pairs are recovered (distance < 12 Å between
        N1/N9 atoms of paired residues).

        This drives the adaptive difficulty loop in the supervised phase.
        """
        self.refiner.eval()
        bp_recoveries = []
        sampled = 0

        for data, name, seqs in loader:
            if sampled >= max_samples:
                break

            frames_true, residue_types, cov_edges, bp_edges, chain_edges = (
                pyg_data_to_kinematic(data, seqs, self.device)
            )
            if bp_edges.shape[1] == 0:
                continue  # skip if no base pairs

            batch = data.batch.to(self.device)
            res_batch = batch[::5]

            frames_init = self.training_scheduler.apply_noise(frames_true)

            if isinstance(seqs, str):
                seqs = [seqs]

            outputs = self.refiner(
                frames=frames_init,
                sequences=seqs,
                residue_types=residue_types,
                batch=res_batch,
                covalent_edges=cov_edges,
                bp_edges=bp_edges,
                chain_edges=chain_edges,
            )

            # Compute BP recovery from predicted coords
            pred_coords = outputs['final_frames'].global_coords()
            reward_dict = self.reward_fn(
                coords=pred_coords,
                residue_types=residue_types,
                bp_edges=bp_edges,
                batch=res_batch,
            )
            bp_recoveries.append(reward_dict['bp_reward'].item())
            sampled += 1

        self.refiner.train()
        return float(np.mean(bp_recoveries)) if bp_recoveries else 0.0

    def _save_checkpoint(self, epoch: int, exp_name: str, phase: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.save_dir, exp_name)
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'gnn_state_dict': self.gnn.state_dict(),
            'lddt_state_dict': self.lddt_suggester.state_dict(),
            'config': vars(self.config),
            'curriculum_scheduler': self.training_scheduler.state_dict(),
        }

        path = os.path.join(save_path, f'{phase}_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.lddt_suggester.load_state_dict(checkpoint['lddt_state_dict'])
        if 'curriculum_scheduler' in checkpoint:
            self.training_scheduler.load_state_dict(
                checkpoint['curriculum_scheduler']
            )
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch {checkpoint['epoch']}, phase {checkpoint['phase']}, "
            f"σ={self.training_scheduler.current_sigma:.4f})"
        )
