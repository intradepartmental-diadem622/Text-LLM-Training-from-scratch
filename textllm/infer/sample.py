"""Turn a row of logits into the next token id.

Supports the usual knobs: temperature, top-k, nucleus (top-p), and a repetition penalty.
Temperature 0 means greedy (argmax), which is handy for deterministic tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_sampling(
    temperature: float, top_k: int | None, top_p: float | None, repetition_penalty: float
) -> None:
    """Reject sampling parameters that would produce nonsense or a confusing crash."""
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0 (0 means greedy), got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if repetition_penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {repetition_penalty}")


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    prev_ids: list[int] | None = None,
) -> int:
    validate_sampling(temperature, top_k, top_p, repetition_penalty)
    logits = logits.float().clone()

    if repetition_penalty != 1.0 and prev_ids:
        for tok in set(prev_ids):
            # Divide positive logits and multiply negative ones — the standard CTRL-style
            # penalty, which pushes already-seen tokens toward being less likely.
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty

    if temperature == 0:
        return int(logits.argmax())

    logits = logits / temperature

    if top_k:
        k = min(top_k, logits.numel())
        threshold = torch.topk(logits, k).values[-1]
        logits[logits < threshold] = float("-inf")

    if top_p:
        ordered, idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1)
        # Keep the smallest set of tokens whose probability mass exceeds top_p.
        cut = cum > top_p
        cut[1:] = cut[:-1].clone()
        cut[0] = False
        logits[idx[cut]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
