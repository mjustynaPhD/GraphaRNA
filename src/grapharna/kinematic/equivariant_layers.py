"""
SE(3)-Equivariant GNN layers for kinematic frame updates.

Implements:
  - EGNN_Layer:     E(n) Equivariant Graph Neural Network layer
  - VN_EGNN_Layer:  Vector-Neuron enhanced EGNN layer with SO(3) output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EGNN_Layer(MessagePassing):
    """E(n)-Equivariant Graph Neural Network layer (Satorras et al., 2021).

    Performs coordinate-equivariant and feature-invariant message passing.
    Updates both scalar features h and vector coordinates x.

    Message:   m_ij = φ_m(h_i, h_j, ||x_i - x_j||², e_ij)
    Coord:     Δx_i = Σ_j (x_i - x_j) · φ_x(m_ij)
    Feature:   h_i' = φ_h(h_i, Σ_j m_ij)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        coord_dim: int = 3,
        act_fn: str = 'silu',
    ):
        super().__init__(aggr='add')
        self.node_dim = node_dim
        self.coord_dim = coord_dim

        # Message MLP:  (h_i, h_j, ||d||², e_ij) → m_ij
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update:  m_ij → scalar weight for (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Node update:  (h_i, aggr_msg) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.msg_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,         # (N, node_dim) node features
        x: torch.Tensor,         # (N, 3) coordinates
        edge_index: torch.Tensor, # (2, E)
        edge_attr: torch.Tensor,  # (E, edge_dim)
    ):
        """
        Returns:
            h_out:  (N, node_dim) updated node features
            x_out:  (N, 3) updated coordinates
        """
        row, col = edge_index
        diff = x[row] - x[col]                         # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True) # (E, 1)

        # Build message input
        msg_input = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)   # (E, hidden_dim)
        msg = self.msg_norm(msg)

        # Coordinate update
        coord_weight = self.coord_mlp(msg)              # (E, 1)
        coord_delta = diff * coord_weight               # (E, 3)
        # Aggregate coordinate deltas per node
        x_agg = torch.zeros_like(x)
        x_agg.scatter_add_(0, row.unsqueeze(-1).expand_as(coord_delta), coord_delta)
        x_out = x + x_agg

        # Message aggregation for node update
        msg_agg = torch.zeros(h.shape[0], msg.shape[-1], device=h.device)
        msg_agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        # Node update
        h_out = self.node_mlp(torch.cat([h, msg_agg], dim=-1))
        h_out = self.node_norm(h_out + h)  # Residual

        return h_out, x_out


class VN_EGNN_Layer(nn.Module):
    """Vector-Neuron enhanced EGNN that outputs SO(3) rotation updates.

    Extends EGNN with vector neurons (Deng et al., 2021) to produce
    equivariant rotation predictions for each residue.

    The key addition is a vector feature track that transforms
    equivariantly under rotations, enabling the network to predict
    rotation matrices via a learned Gram-Schmidt procedure.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        n_vector_features: int = 8,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_vf = n_vector_features

        # Scalar EGNN backbone
        self.egnn = EGNN_Layer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
        )

        # Vector neuron layers: operate on (N, n_vf, 3) tensors
        # Message → vector features
        self.vn_msg_linear = nn.Linear(node_dim, n_vector_features * 3, bias=False)
        # Vector update
        self.vn_update_linear = nn.Linear(n_vector_features, n_vector_features, bias=False)
        self.vn_gate = nn.Sequential(
            nn.Linear(node_dim + n_vector_features, n_vector_features),
            nn.Sigmoid(),
        )

        # SO(3) output heads: predict two vectors for Gram-Schmidt → R
        self.rotation_head = nn.Sequential(
            nn.Linear(node_dim + n_vector_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # Two 3-vectors → Gram-Schmidt → R
        )

        # Translation output head
        self.translation_head = nn.Sequential(
            nn.Linear(node_dim + n_vector_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Confidence output head (pLDDT-like)
        self.confidence_head = nn.Sequential(
            nn.Linear(node_dim + n_vector_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def gram_schmidt(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt orthogonalization to produce rotation matrix.

        Args:
            v1: (N, 3) first vector
            v2: (N, 3) second vector
        Returns:
            R:  (N, 3, 3) rotation matrix
        """
        e1 = F.normalize(v1, dim=-1, eps=1e-8)
        v2_proj = (v2 * e1).sum(-1, keepdim=True) * e1
        e2 = F.normalize(v2 - v2_proj, dim=-1, eps=1e-8)
        e3 = torch.linalg.cross(e1, e2)
        R = torch.stack([e1, e2, e3], dim=-1)  # (N, 3, 3)
        return R

    def forward(
        self,
        h: torch.Tensor,          # (N, node_dim) scalar features
        x: torch.Tensor,          # (N, 3) coordinates (P atom positions)
        v: torch.Tensor,          # (N, n_vf, 3) vector features
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,   # (E, edge_dim)
    ):
        """
        Returns:
            h_out:      (N, node_dim) updated scalar features
            x_out:      (N, 3) updated coordinates
            v_out:      (N, n_vf, 3) updated vector features
            delta_R:    (N, 3, 3) predicted rotation updates
            delta_t:    (N, 3) predicted translation updates
            confidence: (N, 1) per-residue confidence
        """
        N = h.shape[0]

        # 1. Scalar + coordinate update via EGNN
        h_out, x_out = self.egnn(h, x, edge_index, edge_attr)

        # 2. Vector neuron update
        # Generate direction vectors from scalar features
        vn_dirs = self.vn_msg_linear(h_out)  # (N, n_vf*3)
        vn_dirs = vn_dirs.reshape(N, self.n_vf, 3)

        # Vector update with equivariant linear
        v_updated = torch.einsum('ij,njd->nid', self.vn_update_linear.weight, v)
        v_updated = v_updated + vn_dirs

        # Gating: use invariant (norm) features
        v_norms = v_updated.norm(dim=-1)  # (N, n_vf)
        gate_input = torch.cat([h_out, v_norms], dim=-1)
        gate = self.vn_gate(gate_input)   # (N, n_vf)
        v_out = v_updated * gate.unsqueeze(-1)  # (N, n_vf, 3)

        # 3. Predict SE(3) updates
        v_out_norms = v_out.norm(dim=-1)  # (N, n_vf)
        combined = torch.cat([h_out, v_out_norms], dim=-1)

        # Rotation: two vectors → Gram-Schmidt → ΔR
        rot_vecs = self.rotation_head(combined)  # (N, 6)
        v1, v2 = rot_vecs[:, :3], rot_vecs[:, 3:]
        delta_R = self.gram_schmidt(v1, v2)      # (N, 3, 3)

        # Translation
        delta_t = self.translation_head(combined)  # (N, 3)

        # Confidence
        confidence = self.confidence_head(combined)  # (N, 1)

        return h_out, x_out, v_out, delta_R, delta_t, confidence
