"""
LDDT Suggester — predicts per-residue confidence (pLDDT-like) and
identifies fragments for freezing vs. recursive fixing.

Analogous to AlphaFold2's pLDDT but adapted for coarse-grain RNA:
  - Predicts local distance difference test (LDDT) score per residue
  - Classifies residues into "good" (freeze) vs. "needs fixing" (refine)
  - Drives the recursive refinement loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from grapharna.layers import Res


class LDDTSuggester(nn.Module):
    """Predicts per-residue lDDT-like confidence and generates
    masks for recursive refinement.

    Architecture:
        Input:  node features h (from KinematicGNN) + local geometry features
        Output: per-residue pLDDT ∈ [0, 1]
                binary mask: True = needs refinement

    The model learns to predict which residues are poorly placed
    (low confidence) and should be targeted by the next refinement step.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 128,
        n_bins: int = 50,
        confidence_threshold: float = 0.7,
        fragment_expansion: int = 2,
    ):
        """
        Args:
            node_dim:              Input feature dimension (from GNN).
            hidden_dim:            Hidden layer width.
            n_bins:                Number of bins for pLDDT prediction.
            confidence_threshold:  Residues below this score get refined.
            fragment_expansion:    Expand refinement mask by this many
                                   neighbors along the chain (context).
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.fragment_expansion = fragment_expansion
        self.n_bins = n_bins

        # pLDDT prediction network
        self.plddt_net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            Res(hidden_dim),
            Res(hidden_dim),
            nn.Linear(hidden_dim, n_bins),
        )

        # Geometry feature extractor (from local distances)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Clash detector head
        self.clash_detector = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Bin centers for pLDDT computation
        bin_centers = torch.linspace(0.0, 1.0, n_bins)
        self.register_buffer('bin_centers', bin_centers)

    def compute_geometry_features(
        self,
        coords: torch.Tensor,  # (N, 5, 3)
        chain_edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract local geometric features for each residue.

        Features (10-dim):
            - Internal distances: P-C4', P-N, C4'-N, N-C2, C2-C4/C6 (5 features)
            - Sequential distances: P(i)-P(i+1), P(i-1)-P(i) if available (2 features)
            - Bond angles: P-C4'-N angle, C4'-N-C2 angle (2 features)
            - Planarity: deviation of base atoms from plane (1 feature)
        """
        N = coords.shape[0]
        device = coords.device
        features = torch.zeros(N, 10, device=device)

        # Internal distances (within residue)
        P, C4p, N_base, C2, C4_or_C6 = [coords[:, i] for i in range(5)]

        features[:, 0] = (P - C4p).norm(dim=-1)
        features[:, 1] = (P - N_base).norm(dim=-1)
        features[:, 2] = (C4p - N_base).norm(dim=-1)
        features[:, 3] = (N_base - C2).norm(dim=-1)
        features[:, 4] = (C2 - C4_or_C6).norm(dim=-1)

        # Sequential distances
        if chain_edges is not None and chain_edges.shape[1] > 0:
            src, dst = chain_edges
            pp_dist = (P[src] - P[dst]).norm(dim=-1)
            # Forward distance (P(i) → P(i+1))
            features[src, 5] = pp_dist
            # Backward distance (P(i-1) → P(i))
            features[dst, 6] = pp_dist

        # Bond angles (P-C4'-N and C4'-N-C2)
        v1 = P - C4p
        v2 = N_base - C4p
        cos_angle1 = F.cosine_similarity(v1, v2, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        features[:, 7] = torch.acos(cos_angle1)

        v3 = C4p - N_base
        v4 = C2 - N_base
        cos_angle2 = F.cosine_similarity(v3, v4, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        features[:, 8] = torch.acos(cos_angle2)

        # Planarity: distance of C4/C6 from plane defined by N-C2 and perpendicular
        normal = torch.linalg.cross(v4, C4_or_C6 - N_base)  # (N, 3)
        normal = F.normalize(normal, dim=-1, eps=1e-8)
        planarity = (C4_or_C6 - N_base - (C4_or_C6 - N_base) * (normal * (C4_or_C6 - N_base)).sum(-1, keepdim=True) * normal).norm(dim=-1)
        features[:, 9] = planarity

        return features

    def forward(
        self,
        h: torch.Tensor,          # (N, node_dim) node features from GNN
        coords: torch.Tensor,     # (N, 5, 3) current atom coordinates
        chain_edges: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict per-residue confidence and refinement mask.

        Returns:
            dict with keys:
                'plddt':      (N,) predicted pLDDT score ∈ [0, 1]
                'plddt_bins': (N, n_bins) bin logits
                'clash_prob': (N,) probability of steric clash
                'refine_mask': (N,) bool — True = needs refinement
        """
        # Encode local geometry
        geom_features = self.compute_geometry_features(coords, chain_edges)
        geom_emb = self.geometry_encoder(geom_features)  # (N, node_dim)

        # Combine with GNN features
        h_combined = h + geom_emb  # Residual fusion

        # Predict pLDDT
        logits = self.plddt_net(h_combined)  # (N, n_bins)
        probs = F.softmax(logits, dim=-1)
        plddt = (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)  # (N,)

        # Predict clash probability
        clash_prob = self.clash_detector(h_combined).squeeze(-1)  # (N,)

        # Generate refinement mask
        # Refine residues with low confidence OR high clash probability
        low_confidence = plddt < self.confidence_threshold
        high_clash = clash_prob > 0.5
        refine_mask = low_confidence | high_clash

        # Expand mask along chain to include context
        if chain_edges is not None and self.fragment_expansion > 0:
            refine_mask = self._expand_mask(refine_mask, chain_edges)

        return {
            'plddt': plddt,
            'plddt_bins': logits,
            'clash_prob': clash_prob,
            'refine_mask': refine_mask,
        }

    def _expand_mask(
        self,
        mask: torch.Tensor,       # (N,) bool
        chain_edges: torch.Tensor, # (2, E) sequential edges
    ) -> torch.Tensor:
        """Expand refinement mask by `fragment_expansion` hops along chain.

        This ensures that residues adjacent to problematic regions also
        get refined, providing structural context.
        """
        expanded = mask.clone()
        for _ in range(self.fragment_expansion):
            src, dst = chain_edges
            # Propagate mask in both directions
            expanded_new = expanded.clone()
            expanded_new[dst] = expanded_new[dst] | expanded[src]
            expanded_new[src] = expanded_new[src] | expanded[dst]
            expanded = expanded_new
        return expanded

    def compute_lddt_loss(
        self,
        pred_plddt: torch.Tensor,    # (N,) predicted pLDDT
        coords_pred: torch.Tensor,   # (N, 5, 3) predicted coordinates
        coords_true: torch.Tensor,   # (N, 5, 3) ground-truth coordinates
        cutoff: float = 15.0,        # Å — distance cutoff for lDDT
    ) -> torch.Tensor:
        """Compute loss for pLDDT prediction training.

        Ground-truth lDDT is computed by comparing predicted and true
        inter-residue distances at multiple thresholds.
        """
        N = coords_pred.shape[0]
        device = coords_pred.device

        # Compute per-residue lDDT using P atom distances
        pos_pred = coords_pred[:, 0]   # P atoms
        pos_true = coords_true[:, 0]

        # Pairwise distances
        d_pred = torch.cdist(pos_pred, pos_pred)  # (N, N)
        d_true = torch.cdist(pos_true, pos_true)  # (N, N)

        # Only consider pairs within cutoff in the true structure
        pair_mask = (d_true < cutoff) & (~torch.eye(N, device=device).bool())

        # lDDT computation at 4 thresholds: 0.5, 1, 2, 4 Å
        thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device)
        abs_diff = (d_pred - d_true).abs()

        lddt_per_threshold = []
        for t in thresholds:
            conserved = (abs_diff < t).float() * pair_mask.float()
            total = pair_mask.float().sum(dim=-1).clamp(min=1)
            lddt_per_threshold.append(conserved.sum(dim=-1) / total)

        # Average over thresholds → per-residue lDDT
        lddt_true = torch.stack(lddt_per_threshold, dim=0).mean(dim=0)  # (N,)

        # Cross-entropy loss on binned prediction
        loss = F.mse_loss(pred_plddt, lddt_true)
        return loss
