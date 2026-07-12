"""Grouped-query self-attention with a KV-cache.

The fast path is ``scaled_dot_product_attention``, which runs on CPU, MPS, and CUDA and
quietly uses FlashAttention on capable GPUs — so we get one portable, fast implementation
with no per-backend branching. A readable ``_naive`` path is kept alongside it, selected by
``ModelConfig.naive_attention``, purely so the mechanics are legible.

Grouped-query attention lets several query heads share one key/value head, which shrinks
the KV-cache and speeds up decoding at almost no quality cost.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from textllm.model.rope import apply_rotary


class KVCache:
    """Per-layer store of past keys and values, grown one step (or chunk) at a time."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def past_length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, n_kv, T, d) to (B, n_kv * n_rep, T, d) for grouped-query attention."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


class Attention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_kv_head: int, naive: bool = False) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embed // n_head
        self.n_rep = n_head // n_kv_head
        self.naive = naive

        self.wq = nn.Linear(n_embed, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embed, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embed, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_embed, n_embed, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        past = cache.past_length() if cache is not None else 0
        if cache is not None:
            k, v = cache.update(k, v)
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        if self.naive:
            out = self._naive(q, k, v, past)
        elif past == 0:
            # Prefill or training: square, so the built-in causal flag hits the fast path.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=self._decode_mask(q, k))

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)

    def _decode_mask(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Boolean mask so new queries attend to all past keys plus themselves, causally."""
        T, Tk = q.shape[2], k.shape[2]
        past = Tk - T
        query_pos = torch.arange(T, device=q.device)[:, None] + past
        key_pos = torch.arange(Tk, device=q.device)[None, :]
        return key_pos <= query_pos

    def _naive(self, q, k, v, past: int) -> torch.Tensor:
        # The same computation SDPA does, spelled out: scaled dot products, causal mask,
        # softmax, weighted sum of values. Slower, but nothing is hidden.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        T, Tk = q.shape[2], k.shape[2]
        query_pos = torch.arange(T, device=q.device)[:, None] + past
        key_pos = torch.arange(Tk, device=q.device)[None, :]
        scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v
