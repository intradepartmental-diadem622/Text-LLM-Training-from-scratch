"""Pretraining: fit the base model on next-token prediction over a token file."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from textllm.config import Config
from textllm.data import PretrainDataset
from textllm.device import describe, pick_device
from textllm.model import Transformer
from textllm.train.loop import build_optimizer, infinite, load_checkpoint, model_state, train


def run_pretrain(cfg: Config, resume: str | None = None) -> None:
    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    print(f"device: {describe(device)}")

    model = Transformer(cfg.model).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M non-embedding params")
    if cfg.train.compile and device.type != "mps":
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg.train)

    start_step = 0
    if resume:
        ckpt = load_checkpoint(resume, map_location=device)
        # load into the underlying module in case `model` is a torch.compile wrapper
        getattr(model, "_orig_mod", model).load_state_dict(model_state(ckpt))
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {resume} at step {start_step}")

    block = cfg.model.context_length
    data_dir = Path(cfg.train.data_dir)
    # sampling with replacement avoids shuffle's O(N) index permutation on big corpora
    train_set = PretrainDataset(data_dir / "train.bin", block)
    if len(train_set) == 0:
        raise ValueError(
            f"{data_dir / 'train.bin'} has only {len(train_set.data)} tokens — too few to "
            f"form one window of context_length + 1 = {block + 1}; add more data or lower "
            f"model.context_length"
        )
    train_loader = DataLoader(
        train_set,
        batch_size=min(cfg.train.batch_size, len(train_set)),
        sampler=RandomSampler(train_set, replacement=True),
        drop_last=True,
    )
    val_path = data_dir / "val.bin"
    val_batches = None
    if val_path.exists():
        val_set = PretrainDataset(val_path, block)
        if len(val_set) == 0:
            print(f"{val_path} is too short for one window — skipping evaluation")
        else:
            val_loader = DataLoader(
                val_set,
                batch_size=min(cfg.train.batch_size, len(val_set)),
                sampler=RandomSampler(val_set, replacement=True),
                drop_last=True,
            )
            val_batches = infinite(val_loader)

    train(
        model,
        optimizer,
        infinite(train_loader),
        cfg.train,
        device,
        val_batches=val_batches,
        out_dir=cfg.train.out_dir,
        full_config=cfg,
        start_step=start_step,
    )
