"""
Recursive Refiner — the "Self-Correction" loop.

Implements the recursive refinement paradigm:
  1. KinematicGNN predicts structure update (ΔR, Δt)
  2. LDDTSuggester computes per-residue confidence + clash detection
  3. "Bad" sub-graphs (low confidence / clashed) are re-refined
  4. Repeat until convergence or max iterations

Inspired by Recursive Language Models where the model iteratively
corrects its own output.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from grapharna.kinematic.frames import RigidFrame, forward_kinematics
from grapharna.kinematic.kinematic_gnn import KinematicGNN
from grapharna.kinematic.lddt_suggester import LDDTSuggester
from grapharna.kinematic.rl_agent import RewardFunction

logger = logging.getLogger(__name__)


class RecursiveRefiner(nn.Module):
    """Orchestrates the recursive kinematic refinement loop.

    Pipeline per iteration:
        1. Build spatial edges from current 3D graph
        2. Run KinematicGNN → (ΔR, Δt, confidence)
        3. Apply SE(3) updates:  R_new = ΔR · R_old
        4. Run LDDTSuggester → identify poorly-placed residues
        5. If poorly-placed residues exist and steps remain:
           - Mask "good" residues (freeze them)
           - Re-run GNN on the "bad" sub-graph
           - Apply corrections only to bad residues
        6. Return refined structure + confidence scores

    The lever_damping parameter controls the "Lever Effect":
    rotations applied upstream in the kinematic chain are damped
    to prevent catastrophic displacement at the 3' end.
    """

    def __init__(
        self,
        gnn: KinematicGNN,
        lddt_suggester: LDDTSuggester,
        reward_fn: Optional[RewardFunction] = None,
        max_iterations: int = 4,
        convergence_threshold: float = 0.85,
        lever_damping: float = 0.95,
    ):
        super().__init__()
        self.gnn = gnn
        self.lddt_suggester = lddt_suggester
        self.reward_fn = reward_fn
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.lever_damping = lever_damping

    def forward(
        self,
        frames: RigidFrame,
        sequences: list,
        residue_types: torch.Tensor,
        batch: torch.Tensor,
        covalent_edges: Optional[torch.Tensor] = None,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run recursive refinement loop.

        Args:
            frames:           Initial RigidFrame.
            sequences:        List of nucleotide strings.
            residue_types:    (N,) residue type indices.
            batch:            (N,) batch assignment.
            covalent_edges:   (2, E_cov) backbone edges.
            bp_edges:         (2, E_bp) base-pair edges.
            chain_edges:      (2, E_chain) sequential chain edges (for FK).
            return_trajectory: If True, store intermediate structures.

        Returns:
            dict with keys:
                'frames':       Final RigidFrame
                'coords':       (N, 5, 3) final global coordinates
                'plddt':        (N,) final per-residue confidence
                'n_iterations': Number of refinement steps taken
                'trajectory':   List of (coords, plddt) if return_trajectory
                'rewards':      List of reward dicts if reward_fn is provided
        """
        device = frames.device
        N = frames.num_residues

        trajectory = []
        rewards = []
        mask = None  # Initially refine all residues

        for iteration in range(self.max_iterations):
            step = torch.full(
                (N,), iteration, dtype=torch.long, device=device
            )

            # --- Step 1: Run KinematicGNN ---
            gnn_out = self.gnn(
                frames=frames,
                sequences=sequences,
                residue_types=residue_types,
                batch=batch,
                step=step,
                covalent_edges=covalent_edges,
                bp_edges=bp_edges,
                mask=mask,
            )

            delta_R = gnn_out['delta_R']
            delta_t = gnn_out['delta_t']
            h = gnn_out['h']

            # --- Step 2: Apply lever-damped SE(3) update ---
            delta_t_damped = self._apply_lever_damping(
                delta_t, chain_edges, iteration
            )
            frames = frames.apply_update(delta_R, delta_t_damped, mask=mask)

            # --- Step 3: Forward kinematics (optional chain constraint) ---
            coords = frames.global_coords()
            if chain_edges is not None:
                coords = forward_kinematics(
                    frames, chain_edges, self.lever_damping
                )

            # --- Step 4: Compute confidence + identify bad regions ---
            lddt_out = self.lddt_suggester(
                h=h,
                coords=coords,
                chain_edges=chain_edges,
                batch=batch,
            )
            plddt = lddt_out['plddt']
            refine_mask = lddt_out['refine_mask']

            # --- Step 5: Compute reward if available ---
            if self.reward_fn is not None:
                reward_dict = self.reward_fn(
                    frames, bp_edges, chain_edges, batch
                )
                rewards.append(reward_dict)

            # --- Step 6: Store trajectory ---
            if return_trajectory:
                trajectory.append({
                    'coords': coords.detach().clone(),
                    'plddt': plddt.detach().clone(),
                    'mask': mask.detach().clone() if mask is not None else None,
                    'n_refined': refine_mask.sum().item() if refine_mask is not None else N,
                })

            # --- Step 7: Check convergence ---
            mean_plddt = plddt.mean().item()
            n_bad = refine_mask.sum().item() if refine_mask is not None else 0

            logger.debug(
                f"Iteration {iteration}: mean_plddt={mean_plddt:.3f}, "
                f"n_bad={n_bad}/{N}"
            )

            if mean_plddt >= self.convergence_threshold and n_bad == 0:
                logger.info(
                    f"Converged at iteration {iteration} with "
                    f"mean_plddt={mean_plddt:.3f}"
                )
                break

            # --- Step 8: Update mask for next iteration ---
            # Only refine residues that are still "bad"
            mask = refine_mask

            # If no residues need refinement, stop
            if mask is not None and not mask.any():
                break

        return {
            'frames': frames,
            'coords': frames.global_coords(),
            'plddt': plddt,
            'n_iterations': iteration + 1,
            'trajectory': trajectory if return_trajectory else None,
            'rewards': rewards if self.reward_fn is not None else None,
            'h': h,
        }

    def _apply_lever_damping(
        self,
        delta_t: torch.Tensor,     # (N, 3)
        chain_edges: Optional[torch.Tensor],
        iteration: int,
    ) -> torch.Tensor:
        """Apply lever-effect damping to translations.

        For residues deeper in the kinematic chain (farther from 5' end),
        scale down translations to prevent accumulated error.

        The damping factor decreases with:
          - Chain depth (distance from 5' end)
          - Iteration number (later iterations = finer adjustments)
        """
        if chain_edges is None:
            return delta_t

        N = delta_t.shape[0]
        device = delta_t.device

        # Estimate chain depth via BFS from root nodes
        depth = self._compute_chain_depth(N, chain_edges, device)

        # Damping: exponential decay with depth + iteration
        damp_factor = self.lever_damping ** (depth.float() + iteration)
        damp_factor = damp_factor.unsqueeze(-1)  # (N, 1)

        return delta_t * damp_factor

    @staticmethod
    def _compute_chain_depth(
        N: int,
        chain_edges: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute chain depth (distance from 5' root) for each residue."""
        depth = torch.zeros(N, dtype=torch.long, device=device)
        src, dst = chain_edges

        # Find roots (nodes that are sources but never destinations)
        all_src = set(src.cpu().tolist())
        all_dst = set(dst.cpu().tolist())
        roots = all_src - all_dst

        if not roots:
            # Fallback: use first node
            roots = {src[0].item()}

        # BFS
        visited = set()
        queue = [(r, 0) for r in roots]

        while queue:
            node, d = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            depth[node] = d

            # Find children
            children_mask = src == node
            children = dst[children_mask].tolist()
            for child in children:
                if child not in visited:
                    queue.append((child, d + 1))

        return depth

    @torch.no_grad()
    def sample(
        self,
        sequences: list,
        residue_types: torch.Tensor,
        batch: torch.Tensor,
        covalent_edges: Optional[torch.Tensor] = None,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate structure from sequence (inference mode).

        Starts from an unfolded linear chain and iteratively refines.

        Args:
            sequences:     List of nucleotide strings.
            residue_types: (N,) residue type indices.
            batch:         (N,) batch assignment.
            ...

        Returns:
            Same as forward().
        """
        from grapharna.kinematic.frames import initialize_linear_chain

        # Initialize linear chain for each sequence
        all_frames = []
        for seq in sequences:
            frame = initialize_linear_chain(seq, device=device)
            all_frames.append(frame)

        # Concatenate frames across batch
        frames = RigidFrame(
            R=torch.cat([f.R for f in all_frames], dim=0),
            t=torch.cat([f.t for f in all_frames], dim=0),
            local_coords=torch.cat([f.local_coords for f in all_frames], dim=0),
        )

        return self.forward(
            frames=frames,
            sequences=sequences,
            residue_types=residue_types,
            batch=batch,
            covalent_edges=covalent_edges,
            bp_edges=bp_edges,
            chain_edges=chain_edges,
            return_trajectory=True,
        )
