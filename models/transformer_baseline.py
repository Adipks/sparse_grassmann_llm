from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .blocks import LearnedPositionalEmbedding, PositionalEncodingConfig, TransformerBlock


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 432          # matched to Grassmann 21.75M → 21.64M params
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 4 * 432
    max_seq_len: int = 2048
    dropout: float = 0.1
    tie_tok_embeddings: bool = True


class TransformerLM(nn.Module):
    """
    GPT-2 style causal Transformer baseline for language modeling.
    """

    def __init__(self, config: TransformerConfig) -> None:
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
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_tok_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (batch, seq)
        attention_mask: optional additive mask broadcastable to (batch, 1, 1, seq)
        returns: logits (batch, seq, vocab_size)
        """
        x = self.token_embedding(input_ids)  # (b, s, d)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attn_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

