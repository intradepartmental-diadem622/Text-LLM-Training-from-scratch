"""Held-out loss and perplexity for a trained model."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, RandomSampler

from textllm.data import PretrainDataset
from textllm.device import pick_device
from textllm.runtime import load_model


@torch.no_grad()
def perplexity(ckpt_path: str, data_bin: str, iters: int = 200, batch_size: int = 16) -> float:
    device = pick_device()
    model, cfg = load_model(ckpt_path, device)
    dataset = PretrainDataset(data_bin, cfg.context_length)
    if len(dataset) == 0:
        raise ValueError(
            f"{data_bin} is too short to evaluate: it needs more than "
            f"context_length + 1 = {cfg.context_length + 1} tokens"
        )
    # sampling with replacement avoids shuffle's O(N) index permutation on big files
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        sampler=RandomSampler(dataset, replacement=True),
        drop_last=True,
    )

    total, seen = 0.0, 0
    for x, y in loader:
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
        seen += 1
        if seen >= iters:
            break

    mean = total / seen
    return math.exp(mean)
