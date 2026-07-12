"""The full decoder-only transformer.

Backbone: token embedding -> N pre-norm blocks -> final RMSNorm -> tied LM head. Positions
come from RoPE inside attention, so there's no learned position table. ``forward_hidden``
exposes the final hidden state so post-training can bolt on a value or reward head without
duplicating the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from textllm.config import ModelConfig
from textllm.model.attention import KVCache
from textllm.model.block import Block
from textllm.model.norm import RMSNorm
from textllm.model.rope import RotaryEmbedding


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.rope = RotaryEmbedding(cfg.head_dim(), cfg.rope_theta)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.norm = RMSNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        # GPT-2 init: shrink residual projections so variance survives a deep stack
        scale = (2 * cfg.n_blocks) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(
        self,
        idx: torch.Tensor,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T = idx.shape
        positions = torch.arange(start_pos, start_pos + T, device=idx.device)
        cos, sin = self.rope(positions)
        x = self.token_embed(idx)
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = block(x, cos, sin, cache)
        return self.norm(x)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.forward_hidden(idx, caches, start_pos)

        if targets is None:
            # during cached decode only the last position's logits are needed
            logits = self.lm_head(x[:, [-1], :]) if caches is not None else self.lm_head(x)
            return logits, None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1).long(),
            ignore_index=-100,
        )
        return logits, loss

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_weights:
            n -= self.token_embed.weight.numel()
        return n
