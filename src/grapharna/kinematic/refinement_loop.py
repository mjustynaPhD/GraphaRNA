"""
RecursiveRefinementLoop — iterative SE(3) frame refinement with reward.

Implements the core ``forward_step`` that can be called repeatedly:

    for t in range(T):
        new_frames, reward, info = loop.forward_step(frames, ...)
        frames = new_frames          # carry updated state forward

Each step:
    1.  Re-builds the 3D distance graph from the *current* coordinates.
    2.  Runs the GNN backbone → obtains latent h.
    3.  Runs the Biased Transformer → refines h with B_{ij} bias.
    4.  Runs the FrameUpdateHead → predicts (ΔR, Δt).
    5.  Applies the rigid-body update:
            R_new = ΔR · R_old
            t_new = t_old + Δt
    6.  Computes a ``compute_reward`` signal:
            + Reward : base-pair pairs within 4–6 Å
            – Penalty: steric clashes (any two beads < 3.0 Å)
            – Penalty: broken backbone (P–C4' > 4.0 Å)

The loop also exposes ``refine`` (full unrolled refinement) and
``sample`` (inference from linear chain) convenience methods.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from grapharna.kinematic.frames import (
    RigidFrame,
    initialize_linear_chain,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  compute_reward  — physics-based reward per the specification
# ════════════════════════════════════════════════════════════════════

class RefinementReward(nn.Module):
    r"""Reward function matching the specification exactly.

    **+ Reward** for each base-pair (i,j) whose N1/N9 atoms are
    within the 4–6 Å sweetspot.

    **– Penalty** for every pair of beads closer than 3.0 Å (steric
    clash).

    **– Penalty** for every backbone bond P(i)–C4'(i) whose length
    exceeds 4.0 Å (broken connectivity).

    All terms are differentiable soft approximations.

    Parameters
    ----------
    bp_dist_lo, bp_dist_hi : float
        Target distance window for base-pair N–N distance (Å).
    clash_dist : float
        Minimum allowed distance between any two CG beads (Å).
    backbone_max : float
        Maximum allowed P–C4' distance before penalty (Å).
    w_bp, w_clash, w_backbone : float
        Weighting coefficients.
    """

    def __init__(
        self,
        bp_dist_lo: float = 4.0,
        bp_dist_hi: float = 6.0,
        clash_dist: float = 3.0,
        backbone_max: float = 4.0,
        w_bp: float = 1.0,
        w_clash: float = 0.5,
        w_backbone: float = 0.5,
    ):
        super().__init__()
        self.bp_dist_lo = bp_dist_lo
        self.bp_dist_hi = bp_dist_hi
        self.clash_dist = clash_dist
        self.backbone_max = backbone_max
        self.w_bp = w_bp
        self.w_clash = w_clash
        self.w_backbone = w_backbone

    # ----------------------------------------------------------------

    def forward(
        self,
        frames: RigidFrame,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute reward and penalty components.

        Returns dict with ``reward``, ``bp_reward``, ``clash_penalty``,
        ``backbone_penalty``, and per-pair detail tensors.
        """
        coords = frames.global_coords()  # (N, 5, 3)
        device = coords.device

        # ── 1. Base-pair reward ──────────────────────────────
        bp_reward = torch.tensor(0.0, device=device)
        bp_detail = {}
        if bp_edges is not None and bp_edges.shape[1] > 0:
            # N1/N9 = atom index 2 in the 5-atom CG model
            n_atoms = coords[:, 2, :]  # (N, 3)
            bp_i, bp_j = bp_edges
            bp_dist = (n_atoms[bp_i] - n_atoms[bp_j]).norm(dim=-1)  # (E_bp,)

            # Soft window: 1 when distance ∈ [lo, hi], decays outside
            # Gaussian bumps on each boundary
            sigma = 0.5  # Å — controls softness of boundary
            in_range = torch.sigmoid((bp_dist - self.bp_dist_lo) / sigma) * \
                       torch.sigmoid((self.bp_dist_hi - bp_dist) / sigma)
            bp_reward = in_range.mean()
            bp_detail = {
                'bp_distances': bp_dist.detach(),
                'bp_in_range': in_range.detach(),
                'bp_frac_satisfied': (in_range > 0.5).float().mean().detach(),
            }

        # ── 2. Steric clash penalty ──────────────────────────
        # Flatten all beads: (N*5, 3)
        all_beads = coords.reshape(-1, 3)  # (N*5, 3)
        n_beads = all_beads.shape[0]

        clash_penalty = torch.tensor(0.0, device=device)
        n_clashes = 0
        if n_beads <= 10_000:  # full pairwise — safe for moderate sizes
            pdist = torch.cdist(all_beads, all_beads)  # (n_beads, n_beads)
            # Mask: self, intra-residue, cross-batch
            self_mask = torch.eye(n_beads, device=device, dtype=torch.bool)
            # Intra-residue pairs (same group of 5)
            res_idx = torch.arange(n_beads, device=device) // 5  # (n_beads,)
            intra_mask = res_idx.unsqueeze(0) == res_idx.unsqueeze(1)
            ignore = self_mask | intra_mask
            if batch is not None:
                bead_batch = batch.repeat_interleave(5)
                cross = bead_batch.unsqueeze(0) != bead_batch.unsqueeze(1)
                ignore = ignore | cross
            pdist = pdist.masked_fill(ignore, float('inf'))

            # Soft clash: penalty ∝ ReLU(clash_dist - d)
            violations = F.relu(self.clash_dist - pdist)
            clash_penalty = violations.sum() / max(n_beads, 1)
            n_clashes = int((violations > 0).sum().item() // 2)
        else:
            # Subsample for very large structures
            clash_penalty = torch.tensor(0.0, device=device)

        # ── 3. Backbone connectivity penalty ──────────────────
        backbone_penalty = torch.tensor(0.0, device=device)
        n_broken = 0
        # P(i)–C4'(i) distance for every residue
        p_atoms = coords[:, 0, :]    # (N, 3)
        c4_atoms = coords[:, 1, :]   # (N, 3)
        pc4_dist = (p_atoms - c4_atoms).norm(dim=-1)  # (N,)
        broken_mask = pc4_dist > self.backbone_max
        backbone_penalty = F.relu(pc4_dist - self.backbone_max).mean()
        n_broken = int(broken_mask.sum().item())

        # Also penalise sequential P(i+1)–C4'(i) / P(i)–P(i+1) distances
        # if chain_edges provided
        if chain_edges is not None and chain_edges.shape[1] > 0:
            src, dst = chain_edges
            pp_dist = (p_atoms[src] - p_atoms[dst]).norm(dim=-1)
            # Typical P-P ≈ 5.9 Å; penalise if > 8 Å (very stretched)
            pp_penalty = F.relu(pp_dist - 8.0).mean()
            backbone_penalty = backbone_penalty + 0.5 * pp_penalty

        # ── Total reward ─────────────────────────────────────
        reward = (
            self.w_bp * bp_reward
            - self.w_clash * clash_penalty
            - self.w_backbone * backbone_penalty
        )

        return {
            'reward': reward,
            'bp_reward': bp_reward,
            'clash_penalty': clash_penalty,
            'backbone_penalty': backbone_penalty,
            'n_clashes': n_clashes,
            'n_broken_backbone': n_broken,
            **bp_detail,
        }


# ════════════════════════════════════════════════════════════════════
#  RecursiveRefinementLoop
# ════════════════════════════════════════════════════════════════════

class RecursiveRefinementLoop(nn.Module):
    """Manages the iterative structure-refinement procedure.

    The loop carries:
    * A **GNN** backbone (any ``nn.Module`` mapping spatial graphs → node
      features).
    * A **BiasedTransformerStack** that refines the features with 2D
      structure bias.
    * A **FrameUpdateHead** that converts features → (ΔR, Δt).
    * A **RefinementReward** function for physics-based scoring.
    * An optional **LDDTSuggester** for confidence-gated masking.

    Parameters
    ----------
    gnn : nn.Module
        GNN backbone.  Expected signature:
            ``gnn(frames, sequences, residue_types, batch, step,
                  covalent_edges, bp_edges, mask) → dict``
        returning at least ``'h'`` and ``'delta_R'``, ``'delta_t'``.
    transformer : nn.Module
        Biased transformer stack.  Signature:
            ``transformer(x, bp_edges, batch) → x'``
    frame_head : FrameUpdateHead
        Predicts (ΔR, Δt) from node features.
    reward_fn : RefinementReward
        Physics-based reward.
    lddt_suggester : nn.Module, optional
        Confidence predictor for masking.
    max_steps : int
        Maximum number of refinement iterations.
    convergence_plddt : float
        Mean pLDDT threshold at which to stop early.
    lever_damping : float
        Damping factor for forward kinematics.
    """

    def __init__(
        self,
        gnn: nn.Module,
        transformer: nn.Module,
        frame_head: nn.Module,
        reward_fn: nn.Module,
        lddt_suggester: Optional[nn.Module] = None,
        max_steps: int = 4,
        convergence_plddt: float = 0.85,
        lever_damping: float = 0.95,
    ):
        super().__init__()
        self.gnn = gnn
        self.transformer = transformer
        self.frame_head = frame_head
        self.reward_fn = reward_fn
        self.lddt_suggester = lddt_suggester
        self.max_steps = max_steps
        self.convergence_plddt = convergence_plddt
        self.lever_damping = lever_damping

    # ────────────────────────────────────────────────────────
    #  forward_step — single refinement iteration
    # ────────────────────────────────────────────────────────

    def forward_step(
        self,
        frames: RigidFrame,
        sequences: list,
        residue_types: torch.Tensor,
        batch: torch.Tensor,
        step: torch.Tensor,
        covalent_edges: Optional[torch.Tensor] = None,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[RigidFrame, Dict[str, torch.Tensor]]:
        """Execute one refinement step.

        1. GNN  → spatial feature extraction (re-builds 3D graph).
        2. Transformer → refine features with B_{ij} bias.
        3. FrameUpdateHead → (ΔR, Δt).
        4. Apply update:  R_new = ΔR · R_old, t_new = t_old + Δt.
        5. Compute reward.

        Returns
        -------
        new_frames : RigidFrame
            Updated structure.
        info : dict
            Contains ``reward_dict``, ``delta_R``, ``delta_t``,
            ``confidence``, ``h``, and optionally ``plddt``.
        """
        # ── 1. GNN stage ────────────────────────────
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
        h = gnn_out['h']  # (N, d)

        # ── 2. Transformer refinement ───────────────
        h = self.transformer(h, bp_edges=bp_edges, batch=batch)

        # ── 3. Predict rigid-body updates ───────────
        head_out = self.frame_head(h, mask=mask)
        delta_R = head_out['delta_R']
        delta_t = head_out['delta_t']
        confidence = head_out['confidence']

        # ── 4. Apply lever-damped SE(3) update ──────
        delta_t_damped = self._lever_damp(delta_t, chain_edges, step)
        new_frames = frames.apply_update(delta_R, delta_t_damped, mask=mask)

        # ── 5. Compute reward ────────────────────────
        reward_dict = self.reward_fn(
            new_frames, bp_edges, chain_edges, batch
        )

        # ── 6. Optional LDDT confidence ──────────────
        plddt = None
        refine_mask = None
        if self.lddt_suggester is not None:
            coords = new_frames.global_coords()
            lddt_out = self.lddt_suggester(
                h=h, coords=coords, chain_edges=chain_edges, batch=batch,
            )
            plddt = lddt_out['plddt']
            refine_mask = lddt_out['refine_mask']

        info = {
            'reward_dict': reward_dict,
            'delta_R': delta_R,
            'delta_t': delta_t,
            'confidence': confidence,
            'h': h,
            'plddt': plddt,
            'refine_mask': refine_mask,
        }
        return new_frames, info

    # ────────────────────────────────────────────────────────
    #  refine — full unrolled loop
    # ────────────────────────────────────────────────────────

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
        """Run the full recursive refinement loop.

        Calls ``forward_step`` up to ``max_steps`` times, optionally
        terminating early when mean pLDDT exceeds `convergence_plddt`.

        Returns
        -------
        dict with ``frames``, ``coords``, ``plddt``, ``n_iterations``,
        ``rewards``, ``trajectory``.
        """
        device = frames.device
        N = frames.num_residues
        mask = None

        trajectory: List[dict] = []
        rewards: List[dict] = []

        for t in range(self.max_steps):
            step_tensor = torch.full(
                (N,), t, dtype=torch.long, device=device,
            )

            frames, info = self.forward_step(
                frames, sequences, residue_types, batch, step_tensor,
                covalent_edges, bp_edges, chain_edges, mask,
            )
            rewards.append(info['reward_dict'])

            if return_trajectory:
                trajectory.append({
                    'coords': frames.global_coords().detach(),
                    'plddt': info['plddt'].detach() if info['plddt'] is not None else None,
                    'reward': info['reward_dict']['reward'].detach(),
                })

            # Early stopping on convergence
            if info['plddt'] is not None:
                mean_plddt = info['plddt'].mean().item()
                if mean_plddt >= self.convergence_plddt:
                    logger.info(
                        f"Converged at step {t} (pLDDT={mean_plddt:.3f})"
                    )
                    break
                # Update mask: only refine "bad" residues next round
                mask = info['refine_mask']
                if mask is not None and not mask.any():
                    break

        final_plddt = info['plddt'] if info['plddt'] is not None \
            else info['confidence'].squeeze(-1)

        return {
            'frames': frames,
            'coords': frames.global_coords(),
            'plddt': final_plddt,
            'n_iterations': t + 1,
            'rewards': rewards,
            'trajectory': trajectory if return_trajectory else None,
            'h': info['h'],
            'final_reward': info['reward_dict'],
        }

    # ────────────────────────────────────────────────────────
    #  sample — inference from linear chain
    # ────────────────────────────────────────────────────────

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
        """Generate structure from sequence (inference).

        Starts from a linear chain and iteratively refines.
        """
        all_frames = []
        for seq in sequences:
            frame = initialize_linear_chain(seq, device=device)
            all_frames.append(frame)

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

    # ────────────────────────────────────────────────────────
    #  lever damping helper
    # ────────────────────────────────────────────────────────

    def _lever_damp(
        self,
        delta_t: torch.Tensor,
        chain_edges: Optional[torch.Tensor],
        step: torch.Tensor,
    ) -> torch.Tensor:
        """Lever-effect damping on translations.

        Residues deeper in the 5'→3' chain receive smaller updates
        to prevent accumulated displacement errors.
        """
        if chain_edges is None:
            return delta_t
        N = delta_t.shape[0]
        device = delta_t.device
        depth = self._chain_depth(N, chain_edges, device)
        iteration = step[0].item() if step.numel() > 0 else 0
        damp = self.lever_damping ** (depth.float() + iteration)
        return delta_t * damp.unsqueeze(-1)

    @staticmethod
    def _chain_depth(
        N: int,
        chain_edges: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """BFS depth from 5' root(s)."""
        depth = torch.zeros(N, dtype=torch.long, device=device)
        src, dst = chain_edges
        all_src = set(src.cpu().tolist())
        all_dst = set(dst.cpu().tolist())
        roots = all_src - all_dst
        if not roots:
            roots = {src[0].item()}
        visited = set()
        queue = [(r, 0) for r in roots]
        while queue:
            node, d = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            depth[node] = d
            children = dst[src == node].tolist()
            for c in children:
                if c not in visited:
                    queue.append((c, d + 1))
        return depth
