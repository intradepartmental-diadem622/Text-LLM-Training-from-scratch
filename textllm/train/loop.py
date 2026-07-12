"""The training loop every stage shares.

Pretraining, SFT, reward modeling, and DPO all differ only in their data and their loss.
Everything else — the cosine schedule, gradient accumulation, mixed precision, clipping,
checkpointing — lives here and is driven by a ``loss_fn`` callback the caller supplies.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Iterator

import torch
import torch.nn as nn

from textllm.config import Config
from textllm.device import autocast, needs_grad_scaler

Batch = tuple[torch.Tensor, torch.Tensor]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]


def lr_at(step: int, cfg) -> float:
    """Linear warmup, then cosine decay from ``lr`` down to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only, not on norms, biases, or embeddings."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def default_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    x, y = batch
    _, loss = model(x, y)
    return loss


def infinite(loader) -> Iterator:
    """Cycle a DataLoader forever so training is driven by step count, not epochs."""
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise ValueError(
                "DataLoader yielded no batches — is the dataset smaller than batch_size?"
            )


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterator, device, iters: int, loss_fn: LossFn) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):
        batch = _to_device(next(batches), device)
        total += loss_fn(model, batch).item()
    model.train()
    return total / max(1, iters)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterator,
    cfg,
    device,
    *,
    loss_fn: LossFn = default_loss,
    val_batches: Iterator | None = None,
    out_dir: str | Path | None = None,
    full_config: Config | None = None,
    start_step: int = 0,
    on_log: Callable[[dict], None] | None = None,
) -> None:
    scaler = torch.amp.GradScaler(enabled=needs_grad_scaler(device))
    model.train()
    t0 = time.time()

    for step in range(start_step, cfg.steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(cfg.grad_accum):
            batch = _to_device(next(batches), device)
            with autocast(device):
                loss = loss_fn(model, batch) / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % 20 == 0 or step == cfg.steps - 1:
            dt = time.time() - t0
            record = {"step": step, "loss": running, "lr": lr, "elapsed": round(dt, 1)}
            print(f"step {step:>6} | loss {running:.4f} | lr {lr:.2e} | {dt:6.1f}s", flush=True)
            if on_log:
                on_log(record)

        if val_batches is not None and cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            val = evaluate(model, val_batches, device, cfg.eval_iters, loss_fn)
            print(f"  eval @ {step}: val loss {val:.4f}", flush=True)

        if out_dir and cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(Path(out_dir) / f"step_{step}.pt", model, optimizer, step, full_config)
            _prune_checkpoints(out_dir, cfg.keep_last)

    if out_dir:
        save_checkpoint(Path(out_dir) / "final.pt", model, optimizer, cfg.steps, full_config)


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch.to(device, non_blocking=True)


def save_checkpoint(path, model, optimizer, step: int, full_config: Config | None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "step": step,
        "config": full_config.to_dict() if full_config else None,
    }
    torch.save(payload, path)
    print(f"  saved {path}", flush=True)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


def model_state(ckpt: dict) -> dict:
    """Model weights with any torch.compile ``_orig_mod.`` key prefix removed."""
    return {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}


def _prune_checkpoints(out_dir, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(Path(out_dir).glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep_last]:
        old.unlink()
