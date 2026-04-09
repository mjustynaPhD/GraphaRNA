"""
Rigid frame representation for the 5-atom coarse-grain RNA model.

Each residue is represented as a rigid body (SE(3) frame) defined by
its 5 coarse-grain atoms:  P, C4', N1/N9, C2, C4/C6.

The frame is parameterized as (R, t) ∈ SE(3) where:
  - R ∈ SO(3) is a 3×3 rotation matrix
  - t ∈ ℝ³  is the translation (frame origin = P atom position)

Local atom coordinates within each frame are pre-computed from the
training data and remain fixed during kinematic refinement.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Rotation helpers  (all operate on batched [..., 3, 3] or [..., 4] tensors)
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (w, x, y, z) → rotation matrices.

    Args:
        q: (..., 4) unit quaternions.
    Returns:
        R: (..., 3, 3) rotation matrices.
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices → unit quaternions (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Args:
        R: (..., 3, 3) rotation matrices.
    Returns:
        q: (..., 4) unit quaternions.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2  # s = 4*w
    mask1 = trace > 0
    q[mask1, 0] = 0.25 * s[mask1]
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s[mask1]
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s[mask1]
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s[mask1]

    # Case 2: R[0,0] is greatest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-10)) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

    # Case 3: R[1,1] is greatest
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-10)) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

    # Case 4: R[2,2] is greatest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-10)) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    q = F.normalize(q, p=2, dim=-1)
    return q.reshape(*batch_shape, 4)


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle ∈ ℝ³ → rotation matrix via Rodrigues' formula.

    Args:
        axis_angle: (..., 3) — axis direction encodes axis, norm encodes angle.
    Returns:
        R: (..., 3, 3)
    """
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    axis = F.normalize(axis_angle, dim=-1, eps=1e-12)

    # Skew-symmetric matrix K
    kx, ky, kz = axis.unbind(-1)
    zero = torch.zeros_like(kx)
    K = torch.stack([
        zero, -kz,  ky,
        kz,   zero, -kx,
        -ky,  kx,   zero,
    ], dim=-1).reshape(*axis_angle.shape[:-1], 3, 3)

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand_as(K)

    # Rodrigues:  R = I + sin(θ)K + (1 - cos(θ))K²
    R = eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def random_rotation_matrices(n: int, device: torch.device = None) -> torch.Tensor:
    """Sample n uniform random rotation matrices (Haar measure on SO(3))."""
    q = torch.randn(n, 4, device=device)
    q = F.normalize(q, p=2, dim=-1)
    return quaternion_to_rotation_matrix(q)


# ---------------------------------------------------------------------------
# RigidFrame: batched SE(3) frame container
# ---------------------------------------------------------------------------

class RigidFrame:
    """Batched rigid-body frame for residues.

    Stores:
        R:  (N, 3, 3)  rotation matrices
        t:  (N, 3)     translation vectors  (frame origin)
        local_coords:  (N, 5, 3)  atom positions in local frame
                        order: P, C4', N1/N9, C2, C4/C6
    """

    def __init__(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        local_coords: torch.Tensor,
    ):
        assert R.shape[-2:] == (3, 3), f"R must be (..., 3, 3), got {R.shape}"
        assert t.shape[-1] == 3, f"t must be (..., 3), got {t.shape}"
        self.R = R
        self.t = t
        self.local_coords = local_coords  # (N, 5, 3)

    @property
    def device(self):
        return self.R.device

    @property
    def num_residues(self):
        return self.R.shape[0]

    def global_coords(self) -> torch.Tensor:
        """Apply forward kinematics: local → global coordinates.

        Returns:
            coords: (N, 5, 3) atom positions in global frame.
        """
        # x_global = R @ x_local + t
        coords = torch.einsum('nij,nmj->nmi', self.R, self.local_coords) + self.t.unsqueeze(1)
        return coords

    def flat_coords(self) -> torch.Tensor:
        """Return (N*5, 3) flattened global coordinates."""
        return self.global_coords().reshape(-1, 3)

    def apply_update(
        self,
        delta_R: torch.Tensor,
        delta_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> 'RigidFrame':
        """Apply SE(3) update:  R_new = ΔR · R_old,  t_new = t_old + Δt.

        Args:
            delta_R: (N, 3, 3) rotation updates.
            delta_t: (N, 3)   translation updates.
            mask:    (N,) bool — True for residues to update, False to freeze.
        Returns:
            Updated RigidFrame.
        """
        R_new = delta_R @ self.R
        t_new = self.t + delta_t

        if mask is not None:
            mask_R = mask.unsqueeze(-1).unsqueeze(-1).float()
            mask_t = mask.unsqueeze(-1).float()
            R_new = mask_R * R_new + (1 - mask_R) * self.R
            t_new = mask_t * t_new + (1 - mask_t) * self.t

        return RigidFrame(R=R_new, t=t_new, local_coords=self.local_coords)

    def detach(self) -> 'RigidFrame':
        return RigidFrame(
            R=self.R.detach(),
            t=self.t.detach(),
            local_coords=self.local_coords.detach(),
        )

    def clone(self) -> 'RigidFrame':
        return RigidFrame(
            R=self.R.clone(),
            t=self.t.clone(),
            local_coords=self.local_coords.clone(),
        )

    def to(self, device) -> 'RigidFrame':
        return RigidFrame(
            R=self.R.to(device),
            t=self.t.to(device),
            local_coords=self.local_coords.to(device),
        )


# ---------------------------------------------------------------------------
# Initialization: linear chain / unfolded state
# ---------------------------------------------------------------------------

# Ideal inter-residue P-P distance and internal frame geometry (Å)
IDEAL_PP_DISTANCE = 5.9       # mean P(i)-P(i+1) distance
IDEAL_LOCAL_COORDS = {
    # Relative to P as origin, canonical orientation
    # Order: P, C4', N1/N9, C2, C4/C6  (in Å)
    'A': torch.tensor([
        [0.000, 0.000, 0.000],   # P
        [3.908, 1.431, 0.000],   # C4'
        [5.677, -1.277, 0.839],  # N9
        [6.509, -3.307, 1.392],  # C2
        [7.823, -1.029, 0.778],  # C6
    ]),
    'G': torch.tensor([
        [0.000, 0.000, 0.000],
        [3.908, 1.431, 0.000],
        [5.677, -1.277, 0.839],
        [6.509, -3.307, 1.392],
        [7.823, -1.029, 0.778],
    ]),
    'U': torch.tensor([
        [0.000, 0.000, 0.000],
        [3.908, 1.431, 0.000],
        [5.474, -1.356, 0.838],
        [6.344, -3.388, 1.410],
        [7.612, -1.099, 0.771],
    ]),
    'C': torch.tensor([
        [0.000, 0.000, 0.000],
        [3.908, 1.431, 0.000],
        [5.474, -1.356, 0.838],
        [6.344, -3.388, 1.410],
        [7.612, -1.099, 0.771],
    ]),
}


def compute_local_frames_from_coords(
    coords: torch.Tensor,
    residue_types: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SE(3) frames from global CG coordinates.

    Constructs a local frame for each residue using Gram-Schmidt on:
      e1 = C4' - P   (normalized)
      e2 = N - P     (orthogonalized to e1, normalized)
      e3 = e1 × e2

    Args:
        coords: (N, 5, 3) — global CG atom positions per residue.
        residue_types: (N,) int — residue type indices.
    Returns:
        R:  (N, 3, 3)  rotation matrices
        t:  (N, 3)     translations (= P positions)
        local_coords: (N, 5, 3)  atoms in local frame
    """
    P = coords[:, 0]        # (N, 3)
    C4p = coords[:, 1]      # (N, 3)
    N_base = coords[:, 2]   # (N, 3)

    t = P  # frame origin at P

    # Gram-Schmidt orthogonalization
    v1 = C4p - P
    e1 = F.normalize(v1, dim=-1, eps=1e-8)

    v2 = N_base - P
    v2_proj = (v2 * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(v2 - v2_proj, dim=-1, eps=1e-8)

    e3 = torch.linalg.cross(e1, e2)
    e3 = F.normalize(e3, dim=-1, eps=1e-8)

    R = torch.stack([e1, e2, e3], dim=-1)  # (N, 3, 3) — columns are basis vectors

    # Compute local coordinates: x_local = R^T @ (x_global - t)
    centered = coords - t.unsqueeze(1)  # (N, 5, 3)
    local_coords = torch.einsum('nji,nmj->nmi', R, centered)  # R^T @ centered

    return R, t, local_coords


def initialize_linear_chain(
    sequence: str,
    device: torch.device = None,
    randomize_orientations: bool = True,
) -> RigidFrame:
    """Create an initial 'unfolded' linear chain.

    Each residue is placed along the x-axis spaced by IDEAL_PP_DISTANCE
    with internal CG geometry preserved but inter-residue orientations
    randomized.

    Args:
        sequence: nucleotide string, e.g. "AUGCGAU".
        device:   target device.
        randomize_orientations: if True, each frame gets a random SO(3) rotation.
    Returns:
        RigidFrame for the linear chain.
    """
    N = len(sequence)

    # Translations: place P atoms along x-axis
    t = torch.zeros(N, 3, device=device)
    for i in range(N):
        t[i, 0] = i * IDEAL_PP_DISTANCE

    # Local coordinates from ideal geometry
    local_coords = torch.stack([
        IDEAL_LOCAL_COORDS.get(nt, IDEAL_LOCAL_COORDS['A'])
        for nt in sequence
    ]).to(device)  # (N, 5, 3)

    # Frame rotations
    if randomize_orientations:
        R = random_rotation_matrices(N, device=device)
    else:
        R = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1).contiguous()

    return RigidFrame(R=R, t=t, local_coords=local_coords)


def forward_kinematics(
    frames: RigidFrame,
    chain_edges: torch.Tensor,
    lever_damping: float = 0.95,
) -> torch.Tensor:
    """Apply kinematic chain constraints with lever-effect damping.

    Propagates frame transforms along a sequential chain, applying
    exponential damping to prevent small angular changes near the 5' end
    from causing massive displacements at the 3' end.

    Args:
        frames:        RigidFrame with current per-residue frames.
        chain_edges:   (M, 2) sequential backbone edges, sorted 5'→3'.
        lever_damping: damping factor ∈ (0, 1). Effective rotation at
                       depth d is scaled by lever_damping^d.
    Returns:
        coords: (N, 5, 3) global atom coordinates after FK propagation.
    """
    coords = frames.global_coords()  # (N, 5, 3)

    if chain_edges is None or chain_edges.shape[0] == 0:
        return coords

    # Build adjacency: parent → child
    visited = torch.zeros(frames.num_residues, dtype=torch.bool, device=frames.device)
    visited[chain_edges[0, 0]] = True

    for depth, (parent, child) in enumerate(chain_edges):
        parent, child = parent.item(), child.item()
        if visited[child]:
            continue
        visited[child] = True

        # Damped rotation
        damp = lever_damping ** (depth + 1)
        R_parent = frames.R[parent]

        # Child frame in parent's reference
        R_child = frames.R[child]
        t_child = frames.t[child]

        # Apply damped relative transform
        delta_R = R_child @ R_parent.T
        # Damp the angular component via interpolation toward identity
        eye3 = torch.eye(3, device=frames.device)
        delta_R_damped = eye3 + damp * (delta_R - eye3)  # Linear approx for small damp
        # Re-orthogonalize via SVD
        U, _, Vh = torch.linalg.svd(delta_R_damped)
        delta_R_damped = U @ Vh

        # Update child coordinates
        child_local = frames.local_coords[child]  # (5, 3)
        coords[child] = (delta_R_damped @ R_parent) @ child_local.T
        coords[child] = coords[child].T + t_child

    return coords
