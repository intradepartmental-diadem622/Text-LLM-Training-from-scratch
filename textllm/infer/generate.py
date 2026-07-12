"""Autoregressive generation with a KV-cache.

The prompt is run through once to fill the cache (the prefill), then each new token is a
single-token forward pass that reuses everything already computed. That's what makes
decoding scale with sequence length instead of its square.
"""

from __future__ import annotations

from typing import Iterator

import torch

from textllm.infer.sample import sample_next


@torch.no_grad()
def stream(
    model,
    prompt_ids: list[int],
    device,
    *,
    max_new_tokens: int = 128,
    stop_ids: set[int] | None = None,
    **sampling,
) -> Iterator[int]:
    """Yield generated token ids one at a time, stopping early on a stop token."""
    if not prompt_ids:
        raise ValueError("prompt_ids is empty — the model needs at least one token to condition on")
    model.eval()
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    caches = model.new_caches()

    logits, _ = model(idx, caches=caches, start_pos=0)
    pos = idx.shape[1]
    history = list(prompt_ids)

    for _ in range(max_new_tokens):
        next_id = sample_next(logits[0, -1], prev_ids=history, **sampling)
        if stop_ids and next_id in stop_ids:
            break
        history.append(next_id)
        yield next_id
        step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, _ = model(step, caches=caches, start_pos=pos)
        pos += 1


def generate(model, prompt_ids: list[int], device, **kwargs) -> list[int]:
    """Collect a full generation into a list of token ids."""
    return list(stream(model, prompt_ids, device, **kwargs))
