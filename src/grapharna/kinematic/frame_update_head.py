"""
FrameUpdateHead — predicts rigid-body updates for each residue frame.

Output:
    ΔR:  rotation update as a unit quaternion (w,x,y,z) → converted to 3×3
    Δt:  translation update ∈ ℝ³

The model does NOT predict raw Cartesian coordinates.  Instead it predicts
an SE(3) increment that is composed with the current frame:

    R_new  = ΔR · R_old
    t_new  = t_old + Δt

This guarantees that the internal 5-atom coarse-grain geometry of every
residue is *perfectly preserved* across refinement steps.

Two rotation representations are supported:

* **Quaternion** (default) — `output_repr='quat'`
    Network outputs 4-dim vector, L2-normalised → quaternion_to_rotation_matrix
* **6D (Gram-Schmidt)** — `output_repr='6d'`
    Network outputs 6-dim vector, pair of 3-vecs → Gram-Schmidt orthogonalisation

Both are singularity-free and differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal

from grapharna.kinematic.frames import quaternion_to_rotation_matrix


class FrameUpdateHead(nn.Module):
    r"""Predicts per-residue rigid-body (SE(3)) updates.

    Given latent node features ``h`` the head produces:

    * ``delta_R``  (N, 3, 3) — rotation updates (orthogonal matrices)
    * ``delta_t``  (N, 3)    — translation updates
    * ``confidence`` (N, 1)  — per-residue quality estimate (pLDDT-like)

    The rotation is parameterised as either a **unit quaternion** or
    a **6D (Gram-Schmidt)** vector.

    Parameters
    ----------
    node_dim : int
        Dimension of input node features.
    hidden_dim : int
        Width of the internal MLPs.
    output_repr : {'quat', '6d'}
        Rotation representation.  `'quat'` predicts ``(w,x,y,z)`` with
        L2 normalisation; `'6d'` predicts two 3-vectors ortho-normalised
        via Gram-Schmidt.
    translation_scale : float
        The raw Δt is multiplied by this constant (Å).  Keeps the network
        in a numerically friendly regime during early training.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        output_repr: Literal['quat', '6d'] = 'quat',
        translation_scale: float = 1.0,
    ):
        super().__init__()
        self.output_repr = output_repr
        self.translation_scale = translation_scale

        rot_dim = 4 if output_repr == 'quat' else 6

        # ---------- rotation head ----------
        self.rotation_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rot_dim),
        )
        # Initialise rotation head to near-identity
        # For quaternion: bias → (1,0,0,0);  for 6D: bias → (1,0,0, 0,1,0)
        with torch.no_grad():
            self.rotation_mlp[-1].weight.zero_()
            if output_repr == 'quat':
                self.rotation_mlp[-1].bias.copy_(
                    torch.tensor([1.0, 0.0, 0.0, 0.0])
                )
            else:
                self.rotation_mlp[-1].bias.copy_(
                    torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                )

        # ---------- translation head ----------
        self.translation_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # Initialise translation head to zero
        with torch.no_grad():
            self.translation_mlp[-1].weight.zero_()
            self.translation_mlp[-1].bias.zero_()

        # ---------- confidence head ----------
        self.confidence_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    # -----------------------------------------------------------------
    #  Rotation converters
    # -----------------------------------------------------------------

    @staticmethod
    def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """(N, 4) unit-quaternion → (N, 3, 3) rotation matrix."""
        return quaternion_to_rotation_matrix(F.normalize(q, p=2, dim=-1))

    @staticmethod
    def _gram_schmidt(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Two 3-vectors → (N, 3, 3) rotation matrix via Gram-Schmidt."""
        e1 = F.normalize(v1, dim=-1, eps=1e-8)
        v2_proj = (v2 * e1).sum(-1, keepdim=True) * e1
        e2 = F.normalize(v2 - v2_proj, dim=-1, eps=1e-8)
        e3 = torch.linalg.cross(e1, e2)
        return torch.stack([e1, e2, e3], dim=-1)  # (N, 3, 3)

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,  # (N, node_dim)
        mask: Optional[torch.Tensor] = None,  # (N,) bool — True=update
    ) -> Dict[str, torch.Tensor]:
        """Predict per-residue SE(3) updates.

        Returns
        -------
        dict with keys:
            ``delta_R``     (N, 3, 3) rotation update matrices
            ``delta_t``     (N, 3)    translation updates
            ``confidence``  (N, 1)    quality estimate
        """
        N = h.shape[0]
        device = h.device

        # --- Rotation ---
        raw_rot = self.rotation_mlp(h)  # (N, 4) or (N, 6)
        if self.output_repr == 'quat':
            delta_R = self._quat_to_rotmat(raw_rot)
        else:  # 6D
            v1, v2 = raw_rot[:, :3], raw_rot[:, 3:]
            delta_R = self._gram_schmidt(v1, v2)

        # --- Translation ---
        delta_t = self.translation_mlp(h) * self.translation_scale  # (N, 3)

        # --- Confidence ---
        confidence = self.confidence_mlp(h)  # (N, 1)

        # --- Mask: freeze unselected residues (identity rotation, zero translation) ---
        if mask is not None:
            eye = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
            m_r = mask.float().view(N, 1, 1)
            m_t = mask.float().view(N, 1)
            delta_R = m_r * delta_R + (1.0 - m_r) * eye
            delta_t = m_t * delta_t

        return {
            'delta_R': delta_R,
            'delta_t': delta_t,
            'confidence': confidence,
        }
