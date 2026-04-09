"""
KinematicGNN — the core GNN backbone for Recursive Kinematic Refinement.

Architecture (per the refactored specification):

  ┌─────────────────────────────────────────────────────────────┐
  │  1. Input — Multimodal Encoding                             │
  │     v_i = [ RiNALMo_i (1280→d) ‖ f_{2d,i} ]               │
  │     atom features = [ v_i(distributed) ‖ atom_type_emb ]    │
  ├─────────────────────────────────────────────────────────────┤
  │  2. GNN Stage — Spatial EGNN layers on 3D k-NN graph        │
  │     Outputs refined latent h per residue node               │
  ├─────────────────────────────────────────────────────────────┤
  │  3. Transformer Refinement — BiasedTransformerStack          │
  │     Injects B_{ij} attention bias from 2D base pairs        │
  │     Attn(i,j) = softmax( QK^T/√d + B_{ij} )               │
  ├─────────────────────────────────────────────────────────────┤
  │  4. Output — FrameUpdateHead                                │
  │     Predicts ΔR (quaternion or 6D) + Δt per residue        │
  │     Update: R_new = ΔR·R_old, t_new = t_old + Δt           │
  └─────────────────────────────────────────────────────────────┘

The model does *not* predict xyz coordinates.  It predicts rigid-body
updates that perfectly preserve the internal 5-atom CG geometry.
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import knn
from torch_geometric.utils import remove_self_loops
from typing import Optional, Tuple, Dict

from grapharna.layers import MLP, BesselBasisLayer
from grapharna.kinematic.equivariant_layers import VN_EGNN_Layer
from grapharna.kinematic.frames import RigidFrame
from grapharna.kinematic.biased_transformer import BiasedTransformerStack
from grapharna.kinematic.frame_update_head import FrameUpdateHead


class KinematicConfig:
    """Configuration for KinematicGNN."""

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_vector_features: int = 8,
        spatial_knn: int = 16,
        spatial_cutoff: float = 20.0,
        n_radial_basis: int = 16,
        seq_emb_dim: int = 1280,
        num_residue_types: int = 4,
        atoms_per_residue: int = 5,
        max_refinement_steps: int = 4,
        lever_damping: float = 0.95,
        transformer_blocks: int = 4,
        # --- new fields ---
        f2d_dim: int = 8,             # 2D-structure feature dimension
        atom_type_emb_dim: int = 16,  # per-atom-type embedding dim
        n_transformer_heads: int = 8,
        transformer_ffn_dim: int = 512,
        transformer_dropout: float = 0.1,
        bias_lambda: float = 5.0,     # attention-bias strength λ
        rotation_repr: str = 'quat',  # 'quat' or '6d'
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_vector_features = n_vector_features
        self.spatial_knn = spatial_knn
        self.spatial_cutoff = spatial_cutoff
        self.n_radial_basis = n_radial_basis
        self.seq_emb_dim = seq_emb_dim
        self.num_residue_types = num_residue_types
        self.atoms_per_residue = atoms_per_residue
        self.max_refinement_steps = max_refinement_steps
        self.lever_damping = lever_damping
        self.transformer_blocks = transformer_blocks
        self.f2d_dim = f2d_dim
        self.atom_type_emb_dim = atom_type_emb_dim
        self.n_transformer_heads = n_transformer_heads
        self.transformer_ffn_dim = transformer_ffn_dim
        self.transformer_dropout = transformer_dropout
        self.bias_lambda = bias_lambda
        self.rotation_repr = rotation_repr


class SpatialEdgeBuilder(nn.Module):
    """Builds spatial k-NN edges from 3D coordinates.

    Unlike the original PAMNet which uses only sequence-based edges
    augmented with k-NN, this module creates purely distance-based
    spatial edges so the GNN can "see" non-local 3D contacts.
    """

    def __init__(self, k: int, cutoff: float, n_radial_basis: int = 16):
        super().__init__()
        self.k = k
        self.cutoff = cutoff
        self.rbf = BesselBasisLayer(n_radial_basis, cutoff)
        self.edge_encoder = MLP([n_radial_basis + 3, 64])  # RBF + edge_type one-hot

    def forward(
        self,
        pos: torch.Tensor,         # (N, 3)
        batch: torch.Tensor,       # (N,) batch index
        covalent_edges: Optional[torch.Tensor] = None,  # (2, E_cov)
        bp_edges: Optional[torch.Tensor] = None,        # (2, E_bp)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_index: (2, E_total)
            edge_attr:  (E_total, edge_dim)
        """
        # k-NN spatial edges
        row, col = knn(pos, pos, self.k, batch, batch)
        spatial_edges = torch.stack([row, col], dim=0)
        spatial_edges, _ = remove_self_loops(spatial_edges)

        # Distance filter
        dist = (pos[spatial_edges[0]] - pos[spatial_edges[1]]).norm(dim=-1)
        mask = dist <= self.cutoff
        spatial_edges = spatial_edges[:, mask]
        dist = dist[mask]

        # Edge type: 0=spatial, 1=covalent, 2=base-pair
        n_spatial = spatial_edges.shape[1]
        edge_type = torch.zeros(n_spatial, 3, device=pos.device)
        edge_type[:, 0] = 1.0  # spatial edges

        all_edges = [spatial_edges]
        all_dist = [dist]
        all_types = [edge_type]

        if covalent_edges is not None and covalent_edges.shape[1] > 0:
            d_cov = (pos[covalent_edges[0]] - pos[covalent_edges[1]]).norm(dim=-1)
            t_cov = torch.zeros(covalent_edges.shape[1], 3, device=pos.device)
            t_cov[:, 1] = 1.0
            all_edges.append(covalent_edges)
            all_dist.append(d_cov)
            all_types.append(t_cov)

        if bp_edges is not None and bp_edges.shape[1] > 0:
            d_bp = (pos[bp_edges[0]] - pos[bp_edges[1]]).norm(dim=-1)
            t_bp = torch.zeros(bp_edges.shape[1], 3, device=pos.device)
            t_bp[:, 2] = 1.0
            all_edges.append(bp_edges)
            all_dist.append(d_bp)
            all_types.append(t_bp)

        edge_index = torch.cat(all_edges, dim=1)
        all_dist_cat = torch.cat(all_dist, dim=0)
        all_types_cat = torch.cat(all_types, dim=0)

        # Remove duplicate edges
        edge_index, unique_idx = self._unique_edges(edge_index)
        all_dist_cat = all_dist_cat[unique_idx]
        all_types_cat = all_types_cat[unique_idx]

        # Compute edge features
        rbf_feat = self.rbf(all_dist_cat)
        edge_attr = self.edge_encoder(torch.cat([rbf_feat, all_types_cat], dim=-1))

        return edge_index, edge_attr

    def _unique_edges(self, edge_index):
        """Remove duplicate edges, keeping the first occurrence."""
        edge_pairs = edge_index[0] * edge_index.max() + edge_index[1]
        _, unique_idx = torch.unique(edge_pairs, return_inverse=True)
        # Get first occurrence of each unique pair
        first_occ = torch.zeros(unique_idx.max() + 1, dtype=torch.long, device=edge_index.device)
        for i in range(edge_pairs.shape[0] - 1, -1, -1):
            first_occ[unique_idx[i]] = i
        unique_mask = first_occ
        return edge_index[:, unique_mask], unique_mask


class SequenceEncoder(nn.Module):
    """Encodes nucleotide sequences using RiNALMo (frozen).

    This is a simplified version of the original SequenceModule that
    operates at the residue level (1 embedding per residue) instead of
    the atom level.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        from rinalmo.pretrained import get_pretrained_model
        self.rinalmo, self.alphabet = get_pretrained_model(model_name="giga-v1")
        self.projection = nn.Linear(1280, out_dim, bias=False)
        self.act = nn.SiLU()

        # Freeze RiNALMo
        for param in self.rinalmo.parameters():
            param.requires_grad = False

    def forward(self, sequences: list, device: torch.device) -> torch.Tensor:
        """
        Args:
            sequences: list of nucleotide strings.
            device:    target device.
        Returns:
            emb: (total_residues, out_dim) — one embedding per residue.
        """
        self.rinalmo.eval()
        tokens = torch.tensor(
            self.alphabet.batch_tokenize(sequences),
            dtype=torch.int64,
            device=device,
        )
        # RiNALMo tokens include special tokens: positions > 4 are nucleotides
        flat_tokens = tokens.flatten()
        nt_positions = torch.where(flat_tokens > 4)[0]

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.rinalmo(tokens)

        rep = outputs["representation"]  # (B, L, 1280)
        rep = rep.reshape(-1, 1280)       # (B*L, 1280)
        rep = rep[nt_positions]            # (total_residues, 1280)

        emb = self.act(self.projection(rep))
        return emb


class RefinementStepEncoder(nn.Module):
    """Sinusoidal encoding for the refinement step index.

    Analogous to diffusion timestep encoding but for the recursive
    refinement step counter.
    """

    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step: (N,) refinement step index per residue.
        Returns:
            emb: (N, dim) step embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=step.device).float()
            * -(math.log(100.0) / (half - 1))
        )
        args = step.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


class SecondaryStructureEncoder(nn.Module):
    """Encodes local 2D structure information per residue.

    For each residue *i* produces a small feature vector ``f_{2d,i}``:
        [is_paired (1), pairing_degree (1), loop_type one-hot (6)]

    This is derived from the base-pair edges and chain topology.
    """

    def __init__(self, out_dim: int = 8):
        super().__init__()
        # Raw 2D features: is_paired (1) + pairing_degree (1) + loop_type (6) = 8
        self.proj = nn.Sequential(
            nn.Linear(8, out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        N: int,
        bp_edges: Optional[torch.Tensor],  # (2, E_bp)
        chain_edges: Optional[torch.Tensor],  # (2, E_chain)
        device: torch.device,
    ) -> torch.Tensor:
        """Return (N, out_dim) 2D-structure features per residue."""
        raw = torch.zeros(N, 8, device=device)

        if bp_edges is not None and bp_edges.shape[1] > 0:
            src, dst = bp_edges
            # is_paired flag
            paired = torch.zeros(N, device=device)
            paired[src] = 1.0
            paired[dst] = 1.0
            raw[:, 0] = paired

            # pairing degree (how many partners)
            degree = torch.zeros(N, device=device)
            degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
            degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            raw[:, 1] = degree.clamp(max=5) / 5.0  # normalise

            # Loop-type heuristic (simplified):
            #   0: unpaired, 1: stem, 2: hairpin, 3: internal,
            #   4: bulge, 5: multi-loop
            # Full classification requires SS parsing; here we use a
            # lightweight proxy based on local connectivity.
            raw[:, 2] = (paired == 0).float()          # unpaired
            raw[:, 3] = (degree >= 1).float()           # in a stem
            # Residues paired but flanked by unpaired → likely hairpin
            if chain_edges is not None and chain_edges.shape[1] > 0:
                csrc, cdst = chain_edges
                has_unpaired_neighbor = torch.zeros(N, device=device)
                unpaired_mask = (paired == 0).float()
                has_unpaired_neighbor.scatter_add_(
                    0, csrc, unpaired_mask[cdst]
                )
                raw[:, 4] = ((degree >= 1) & (has_unpaired_neighbor > 0)).float()

        return self.proj(raw)  # (N, out_dim)


class KinematicGNN(nn.Module):
    """Core GNN backbone for Recursive Kinematic Refinement.

    Architecture:
        1. **Multimodal Input Encoding**
           – RiNALMo embeddings (1280-dim, frozen) projected to node_dim
           – 2D structure features ``f_{2d}``
           – Residue-type + refinement-step embeddings
           – Node feature:  v_i = [ RiNALMo_i ‖ f_{2d,i} ]

        2. **GNN Stage** — stacked VN-EGNN layers on the 3D
           spatial neighbourhood graph.

        3. **Transformer Refinement Stage** — ``BiasedTransformerStack``
           with B_{ij} attention bias from secondary structure.

        4. **Output** — The model returns refined latent ``h``.
           ΔR / Δt / confidence are produced by the external
           ``FrameUpdateHead``  (composed in the refinement loop).

    The model operates at the *residue level*: each node is one
    nucleotide frame.
    """

    def __init__(self, config: KinematicConfig):
        super().__init__()
        self.config = config

        # ────────── 1. Multimodal Input Encoding ──────────

        #  a) RiNALMo sequence embeddings (frozen)  → node_dim
        self.seq_encoder = SequenceEncoder(out_dim=config.node_dim)

        #  b) 2D secondary-structure features → f2d_dim
        self.ss_encoder = SecondaryStructureEncoder(out_dim=config.f2d_dim)

        #  c) Residue-type embedding
        self.residue_emb = nn.Embedding(config.num_residue_types, config.node_dim)

        #  d) Refinement step encoding
        self.step_encoder = RefinementStepEncoder(dim=16)

        #  e) Combine: [RiNALMo ‖ f_{2d}] + res_emb + step → node_dim
        input_cat_dim = config.node_dim + config.f2d_dim + config.node_dim + 16
        self.input_projection = MLP([input_cat_dim, config.node_dim])

        # Vector feature initialisation (for VN-EGNN)
        self.vn_init = nn.Linear(
            config.node_dim, config.n_vector_features * 3, bias=False,
        )

        # ────────── 2. GNN Stage ──────────

        self.edge_builder = SpatialEdgeBuilder(
            k=config.spatial_knn,
            cutoff=config.spatial_cutoff,
            n_radial_basis=config.n_radial_basis,
        )

        self.gnn_layers = nn.ModuleList([
            VN_EGNN_Layer(
                node_dim=config.node_dim,
                edge_dim=64,  # SpatialEdgeBuilder output
                hidden_dim=config.hidden_dim,
                n_vector_features=config.n_vector_features,
            )
            for _ in range(config.n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.node_dim)
            for _ in range(config.n_layers)
        ])

        # ────────── 3. Transformer Refinement Stage ──────────

        self.transformer = BiasedTransformerStack(
            d_model=config.node_dim,
            n_heads=config.n_transformer_heads,
            n_layers=config.transformer_blocks,
            ffn_dim=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            bias_lambda=config.bias_lambda,
        )

        # ────────── 4. Output — FrameUpdateHead ──────────

        self.frame_head = FrameUpdateHead(
            node_dim=config.node_dim,
            hidden_dim=config.hidden_dim,
            output_repr=config.rotation_repr,
        )

    def forward(
        self,
        frames: RigidFrame,
        sequences: list,
        residue_types: torch.Tensor,        # (N,) int
        batch: torch.Tensor,                # (N,) batch index
        step: torch.Tensor,                 # (N,) refinement step
        covalent_edges: Optional[torch.Tensor] = None,
        bp_edges: Optional[torch.Tensor] = None,
        chain_edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,  # (N,) bool — True=refine
    ) -> Dict[str, torch.Tensor]:
        """Single refinement step of KinematicGNN.

        The model does **not** predict xyz coordinates.  It predicts a
        rigid-body update (ΔR, Δt) per residue frame via:

        1.  Multimodal encoding:  v_i = [RiNALMo_i ‖ f_{2d,i}]
        2.  GNN on the 3D spatial graph → refined h.
        3.  Biased Transformer (B_{ij}) → globally-refined h.
        4.  FrameUpdateHead → (ΔR, Δt, confidence).

        Args:
            frames:         Current RigidFrame for all residues.
            sequences:      List of nucleotide strings (per-structure).
            residue_types:  (N,) residue type indices (A=0, G=1, U=2, C=3).
            batch:          (N,) batch assignment.
            step:           (N,) current refinement step index.
            covalent_edges: (2, E_cov) backbone covalent edges.
            bp_edges:       (2, E_bp) base-pair edges.
            chain_edges:    (2, E_chain) sequential chain edges.
            mask:           (N,) bool — which residues to refine (None=all).

        Returns:
            dict with keys:
                ``delta_R``     (N, 3, 3) rotation updates
                ``delta_t``     (N, 3)    translation updates
                ``confidence``  (N, 1)    per-residue confidence
                ``h``           (N, node_dim) refined node features
        """
        device = frames.device
        N = frames.num_residues

        # ────────── 1. Multimodal Input Encoding ──────────

        # a) RiNALMo 1280-dim → node_dim
        seq_emb = self.seq_encoder(sequences, device)  # (N, node_dim)
        assert seq_emb.shape[0] == N, \
            f"Sequence embeddings ({seq_emb.shape[0]}) != residues ({N})"

        # b) 2D secondary-structure features
        f2d = self.ss_encoder(N, bp_edges, chain_edges, device)  # (N, f2d_dim)

        # c) Residue-type embedding
        res_emb = self.residue_emb(residue_types)  # (N, node_dim)

        # d) Refinement-step encoding
        step_emb = self.step_encoder(step)  # (N, 16)

        # e) Concatenate & project: [RiNALMo_i ‖ f_{2d,i} ‖ res_emb ‖ step_emb]
        h = self.input_projection(
            torch.cat([seq_emb, f2d, res_emb, step_emb], dim=-1)
        )  # (N, node_dim)

        # ────────── 2. GNN Stage ──────────

        # Initialise vector features for VN-EGNN
        v = self.vn_init(h).reshape(N, self.config.n_vector_features, 3)

        # P-atom positions as node coordinates
        pos = frames.t.clone()  # (N, 3)

        # Build spatial + structural edges (re-computed every step!)
        edge_index, edge_attr = self.edge_builder(
            pos, batch, covalent_edges, bp_edges,
        )

        for layer, ln in zip(self.gnn_layers, self.layer_norms):
            if N > 300:  # gradient checkpointing for large graphs
                h, pos, v, _dr, _dt, _c = checkpoint(
                    layer, h, pos, v, edge_index, edge_attr,
                    use_reentrant=False,
                )
            else:
                h, pos, v, _dr, _dt, _c = layer(
                    h, pos, v, edge_index, edge_attr,
                )
            h = ln(h)

        # ────────── 3. Transformer Refinement ──────────
        # Injects B_{ij} attention bias from 2D base pairs
        h = self.transformer(h, bp_edges=bp_edges, batch=batch)

        # ────────── 4. Frame Update Head ──────────
        head_out = self.frame_head(h, mask=mask)

        return {
            'delta_R': head_out['delta_R'],
            'delta_t': head_out['delta_t'],
            'confidence': head_out['confidence'],
            'h': h,
        }

    def fine_tuning(self):
        """Freeze sequence encoder, train only GNN + heads."""
        for param in self.seq_encoder.parameters():
            param.requires_grad = False
        for param in self.gnn_layers.parameters():
            param.requires_grad = True
