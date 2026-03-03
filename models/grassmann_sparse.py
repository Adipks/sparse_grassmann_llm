from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

from .blocks import GrassmannBlock, LearnedPositionalEmbedding, PositionalEncodingConfig
from utils.sparsity import apply_2to4_masks


@dataclass
class GrassmannConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 6
    d_ff: int = 4 * 512
    reduced_dim: int = 32                      # Paper uses r=32
    # Larger windows suit longer contexts: covers ~2048 / 6 ≈ 340 tokens per layer
    window_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    max_seq_len: int = 2048
    dropout: float = 0.1
    tie_tok_embeddings: bool = True
    apply_sparse: bool = True                  # Set False for dense ablation


class GrassmannLM(nn.Module):
    """
    Causal language model based on Grassmann mixing (paper-exact implementation).
    """

    def __init__(self, config: GrassmannConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = LearnedPositionalEmbedding(
            PositionalEncodingConfig(
                max_seq_len=config.max_seq_len,
                d_model=config.d_model,
            )
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                GrassmannBlock(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    reduced_dim=config.reduced_dim,
                    window_sizes=config.window_sizes,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_tok_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Weight initialisation matching the original repo
        self.apply(self._init_weights)

        # Apply 2:4 structured sparsity masks to all MaskedLinear layers
        # (FFN fc1/fc2 and Grassmann W_red, W_plu, W_gate)
        if config.apply_sparse:
            apply_2to4_masks(self)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq)
        returns: logits (batch, seq, vocab_size)
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

