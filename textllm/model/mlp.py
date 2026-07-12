"""SwiGLU feed-forward network.

A gated variant of the usual two-layer MLP: one projection is passed through SiLU and
used to gate the other. It consistently beats a plain ReLU MLP at the same parameter
count, which is why the hidden size is trimmed to ~2/3 of the classic 4x expansion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, n_embed: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(n_embed, hidden, bias=False)
        self.up = nn.Linear(n_embed, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
