"""A single pre-norm transformer block: attention then feed-forward, each residual."""

from __future__ import annotations

import torch
import torch.nn as nn

from textllm.config import ModelConfig
from textllm.model.attention import Attention, KVCache
from textllm.model.mlp import SwiGLU
from textllm.model.norm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embed)
        hidden = (hidden + 63) // 64 * 64  # round to a multiple of 64 for tidy matmuls
        self.attn_norm = RMSNorm(cfg.n_embed)
        self.attn = Attention(cfg.n_embed, cfg.n_head, cfg.n_kv_head, cfg.naive_attention)
        self.mlp_norm = RMSNorm(cfg.n_embed)
        self.mlp = SwiGLU(cfg.n_embed, hidden)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x
