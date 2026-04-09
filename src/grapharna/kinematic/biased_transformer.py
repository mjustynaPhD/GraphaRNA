"""
BiasedTransformerLayer — Transformer with 2D secondary-structure attention bias.

The standard multi-head attention is augmented with an additive bias
matrix **B** derived from the 2D structure input:

    Attn(i,j) = Softmax( Q K^T / √d  +  B_{ij} )

where  B_{ij} = λ   if residues i,j form a Watson-Crick / non-canonical
                      base pair in the secondary structure,
               = 0   otherwise.

This injects the secondary-structure constraint *directly* into the
attention weights, ensuring that paired residues attend to each other
regardless of sequence distance.

The layer is designed as a drop-in refinement stage after the GNN:

    GNN output h  →  BiasedTransformerLayer(h, bias)  →  h'

Multiple such layers can be stacked.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ────────────────────────────────────────────────────────────────────
#  Utility: build the bias matrix from base-pair edges
# ────────────────────────────────────────────────────────────────────

def build_bp_bias_matrix(
    bp_edges: torch.Tensor,   # (2, E_bp) paired residue indices
    N: int,                    # total number of residues
    lam: float = 5.0,         # additive bias strength λ
    batch: Optional[torch.Tensor] = None,  # (N,) batch assignment
    device: torch.device = None,
) -> torch.Tensor:
    r"""Construct the base-pair attention bias matrix **B** ∈ ℝ^{N×N}.

    B_{ij} = λ  if (i,j) or (j,i) is in ``bp_edges``
    B_{ij} = 0  otherwise

    Batched structures: B is block-diagonal; off-structure entries are
    set to ``-inf`` so that attention cannot leak between structures.

    Parameters
    ----------
    bp_edges : (2, E_bp) tensor
        Source/target residue pairs for base-pairs.
    N : int
        Total number of residue nodes in the batch.
    lam : float
        Bias magnitude for paired residues.
    batch : (N,) int tensor, optional
        Batch assignment.  If provided, cross-structure entries are masked.
    device : torch.device, optional

    Returns
    -------
    B : (N, N) float tensor — additive attention bias.
    """
    if device is None:
        device = bp_edges.device

    B = torch.zeros(N, N, device=device)

    if bp_edges is not None and bp_edges.shape[1] > 0:
        src, dst = bp_edges[0], bp_edges[1]
        B[src, dst] = lam
        B[dst, src] = lam  # symmetric

    # Block-diagonal masking for batched graphs
    if batch is not None:
        cross_mask = batch.unsqueeze(0) != batch.unsqueeze(1)  # (N, N)
        B = B.masked_fill(cross_mask, float('-inf'))

    return B


# ────────────────────────────────────────────────────────────────────
#  BiasedMultiHeadAttention
# ────────────────────────────────────────────────────────────────────

class BiasedMultiHeadAttention(nn.Module):
    """Multi-head self-attention with an additive pairwise bias.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention-weight dropout probability.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,             # (N, d_model)
        bias: Optional[torch.Tensor] = None,  # (N, N)
        key_padding_mask: Optional[torch.Tensor] = None,  # (N,) bool
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        x : (N, d_model)
            Input node features.
        bias : (N, N) optional
            Additive bias added to raw attention scores *before* softmax.
            B_{ij} = λ for paired residues, 0 otherwise, -inf for
            cross-structure masking.
        key_padding_mask : (N,) bool, optional
            True for positions to **ignore** (e.g. padding).

        Returns
        -------
        out : (N, d_model)
        """
        N = x.size(0)
        H, dk = self.n_heads, self.d_k

        Q = self.W_q(x).view(N, H, dk).transpose(0, 1)  # (H, N, dk)
        K = self.W_k(x).view(N, H, dk).transpose(0, 1)  # (H, N, dk)
        V = self.W_v(x).view(N, H, dk).transpose(0, 1)  # (H, N, dk)

        # Scaled dot-product:  (H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)

        # ---- Inject 2D structure bias ----
        if bias is not None:
            # bias is (N, N); broadcast across heads
            scores = scores + bias.unsqueeze(0)  # (H, N, N)

        if key_padding_mask is not None:
            # mask shape: (N,) → (1, 1, N)
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(0).unsqueeze(1), float('-inf')
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (H, N, dk)
        out = out.transpose(0, 1).contiguous().view(N, self.d_model)
        return self.W_o(out)


# ────────────────────────────────────────────────────────────────────
#  BiasedTransformerLayer
# ────────────────────────────────────────────────────────────────────

class BiasedTransformerLayer(nn.Module):
    r"""Single Transformer encoder layer with 2D-structure attention bias.

    Architecture::

        x  →  LN  →  BiasedMHA(+B)  →  +x  →  LN  →  FFN  →  +x

    Parameters
    ----------
    d_model : int
        Feature dimension.
    n_heads : int
        Number of attention heads.
    ffn_dim : int
        Hidden dimension of the feed-forward network.
    dropout : float
        Dropout on attention weights and FFN.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BiasedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                      # (N, d_model)
        bias: Optional[torch.Tensor] = None,   # (N, N)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one biased-transformer block.

        Returns (N, d_model) refined features.
        """
        # Pre-LN self-attention + residual
        x = x + self.attn(self.norm1(x), bias=bias, key_padding_mask=key_padding_mask)
        # Pre-LN FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


class BiasedTransformerStack(nn.Module):
    """Stack of ``n_layers`` BiasedTransformerLayers.

    Convenience wrapper that handles the base-pair bias matrix
    construction and repeated application.

    Parameters
    ----------
    d_model : int
        Node feature dimension.
    n_heads : int
    n_layers : int
    ffn_dim : int
    dropout : float
    bias_lambda : float
        Additive attention bias strength λ for paired residues.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        bias_lambda: float = 5.0,
    ):
        super().__init__()
        self.bias_lambda = bias_lambda
        self.layers = nn.ModuleList([
            BiasedTransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # (N, d_model)
        bp_edges: Optional[torch.Tensor] = None,  # (2, E_bp)
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full transformer stack with 2D-structure bias.

        Returns (N, d_model).
        """
        N = x.size(0)

        # Build additive bias matrix once
        if bp_edges is not None and bp_edges.shape[1] > 0:
            bias = build_bp_bias_matrix(
                bp_edges, N, lam=self.bias_lambda,
                batch=batch, device=x.device,
            )
        else:
            # No base-pair info → still need block-diagonal mask for batching
            bias = None
            if batch is not None:
                cross = batch.unsqueeze(0) != batch.unsqueeze(1)
                bias = torch.zeros(N, N, device=x.device)
                bias = bias.masked_fill(cross, float('-inf'))

        for layer in self.layers:
            x = layer(x, bias=bias)

        return self.final_norm(x)
