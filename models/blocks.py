from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    """
    Linear layer with an elementwise multiplicative mask on weights.

    The mask is registered as a non-trainable buffer. By default it is all ones
    (dense). In the sparsity step we will overwrite it with a 2:4 structured
    mask and optionally convert to semi-structured sparse kernels for inference.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        mask = torch.ones_like(self.weight)
        self.register_buffer("weight_mask", mask, persistent=False)

    def set_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.weight.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match weight {self.weight.shape}")
        self.weight_mask.data.copy_(mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        masked_weight = self.weight * self.weight_mask
        return F.linear(input, masked_weight, self.bias)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = MaskedLinear(d_model, d_ff)
        self.fc2 = MaskedLinear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq, d_model)
        attn_mask: optional additive mask broadcastable to (batch, n_heads, seq, seq)
        """
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shape into heads.
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal masking.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Causal mask: no access to future tokens.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class PluckerEncoder(nn.Module):
    """
    Plucker coordinate encoder per paper (arXiv 2512.19428).

    p_ij^(delta)(t) = z_{t,i} * z_{t-delta,j} - z_{t,j} * z_{t-delta,i}

    With L2 normalization: p_hat = p / max(||p||_2, eps)
    """

    def __init__(self, reduced_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.reduced_dim = reduced_dim
        self.eps = eps
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Pre-compute upper-triangular index pairs.
        indices_i, indices_j = [], []
        for i in range(reduced_dim):
            for j in range(i + 1, reduced_dim):
                indices_i.append(i)
                indices_j.append(j)
        self.register_buffer("idx_i", torch.tensor(indices_i, dtype=torch.long))
        self.register_buffer("idx_j", torch.tensor(indices_j, dtype=torch.long))

    def forward(self, z_current: torch.Tensor, z_past: torch.Tensor) -> torch.Tensor:
        """
        Compute L2-normalized Plucker coordinates.

        Args:
            z_current: (batch, seq, r) current position vectors
            z_past:    (batch, seq, r) past position vectors
        Returns:
            L2-normalized Plucker coordinates (batch, seq, plucker_dim)
        """
        # p_ij = z_current_i * z_past_j - z_current_j * z_past_i
        p = (z_current[..., self.idx_i] * z_past[..., self.idx_j]
             - z_current[..., self.idx_j] * z_past[..., self.idx_i])

        # L2 normalize: p_hat = p / max(||p||_2, eps)
        norm = torch.norm(p, dim=-1, keepdim=True)
        p_normalized = p / torch.clamp(norm, min=self.eps)
        return p_normalized


class GrassmannMixing(nn.Module):
    """
    Causal Grassmann Mixing layer — exact paper implementation (arXiv 2512.19428).

    Paper's forward pass:
    1. z_t = W_red * h_t + b_red
    2. For each delta: p_ij = z_t_i * z_{t-delta}_j - z_t_j * z_{t-delta}_i
    3. p_hat = p / max(||p||_2, eps)   [L2 normalize]
    4. g_t^(delta) = W_plu * p_hat + b_plu  [project each window]
    5. g_t = average(g_t^(delta)) across valid deltas
    6. alpha = sigmoid(W_gate * [h_t; g_t] + b_gate)  [gate from concat]
    7. h_mix = alpha * h_t + (1-alpha) * g_t  [blend, not add]
    8. Apply LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        reduced_dim: int = 32,
        window_sizes: Optional[list] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.reduced_dim = reduced_dim
        # Paper: {1, 2, 4, 8, 12, 16} for 6-layer
        self.window_sizes = window_sizes or [1, 2, 4, 8, 12, 16]
        self.plucker_dim = reduced_dim * (reduced_dim - 1) // 2

        # Step 1: Linear reduction  z = W_red * h + b_red
        self.W_red = nn.Linear(d_model, reduced_dim)

        # Plucker encoder with L2 normalization
        self.plucker = PluckerEncoder(reduced_dim)

        # Step 4: Project Plucker to model dim
        self.W_plu = nn.Linear(self.plucker_dim, d_model)

        # Step 6: Gate from concatenated [h; g]  (2*d_model -> d_model)
        self.W_gate = nn.Linear(2 * d_model, d_model)

        # Step 8: LayerNorm after mixing
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_red.weight)
        nn.init.zeros_(self.W_red.bias)
        nn.init.xavier_uniform_(self.W_plu.weight)
        nn.init.zeros_(self.W_plu.bias)
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Causal Grassmann Mixing per paper.

        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            Mixed hidden states (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Step 1: Reduce to low dimension
        z = self.W_red(hidden_states)  # (batch, seq_len, reduced_dim)

        # Steps 2-5: Compute Plucker features for all windows, then average
        geo_accum = torch.zeros(batch_size, seq_len, self.d_model, device=device, dtype=dtype)
        counts = torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype)

        for delta in self.window_sizes:
            if delta >= seq_len:
                continue

            # Causal: current looks back by delta
            z_current = z[:, delta:, :]   # (batch, seq_len-delta, r)
            z_past = z[:, :-delta, :]     # (batch, seq_len-delta, r)

            # Steps 2-3: Plucker coordinates with L2 normalization
            p_hat = self.plucker(z_current, z_past)  # (batch, seq_len-delta, plucker_dim)

            # Step 4: Project to model dim
            g_delta = self.W_plu(p_hat)  # (batch, seq_len-delta, d_model)

            # Accumulate for averaging
            geo_accum[:, delta:, :] = geo_accum[:, delta:, :] + g_delta
            counts[:, delta:, :] = counts[:, delta:, :] + 1

        # Step 5: Average across valid windows
        counts = counts.clamp(min=1)
        g = geo_accum / counts  # (batch, seq_len, d_model)

        # Step 6: Gating — concatenate [h; g] then sigmoid
        concat = torch.cat([hidden_states, g], dim=-1)  # (batch, seq_len, 2*d_model)
        alpha = torch.sigmoid(self.W_gate(concat))       # (batch, seq_len, d_model)

        # Step 7: Blend (not add!) — h_mix = alpha * h + (1-alpha) * g
        h_mix = alpha * hidden_states + (1 - alpha) * g

        # Step 8: LayerNorm and dropout
        output = self.layer_norm(h_mix)
        output = self.dropout(output)
        return output


class GrassmannBlock(nn.Module):
    """
    Transformer block with Causal Grassmann Mixing.
    Structure: LN -> Grassmann -> Residual -> LN -> FFN -> Residual
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        reduced_dim: int = 32,
        window_sizes: Optional[list] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.grassmann = GrassmannMixing(
            d_model=d_model,
            reduced_dim=reduced_dim,
            window_sizes=window_sizes,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Grassmann mixing with residual
        x = x + self.grassmann(self.ln1(x))
        # FFN with residual
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class PositionalEncodingConfig:
    max_seq_len: int
    d_model: int


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, config: PositionalEncodingConfig) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.embedding = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, d_model) — only seq length is used.
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq)
        pos_emb = self.embedding(positions)  # (1, seq, d_model)
        return x + pos_emb

