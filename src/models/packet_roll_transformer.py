from __future__ import annotations

import torch
from torch import nn


class PacketRollTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_context_length: int = 30,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_context_length, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(model_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(x) + self.position_embedding(positions)
        encoded = self.encoder(hidden)
        last_hidden = self.dropout(self.norm(encoded[:, -1, :]))
        return self.output(last_hidden)
