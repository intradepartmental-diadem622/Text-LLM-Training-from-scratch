"""Supervised fine-tuning: teach the base model to answer in the chat format."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from textllm.chat_template import PAD, Message
from textllm.config import Config, ModelConfig
from textllm.data import SFTDataset, collate_lm
from textllm.device import describe, pick_device
from textllm.model import Transformer
from textllm.tokenizer import get_tokenizer
from textllm.train.loop import build_optimizer, infinite, load_checkpoint, model_state, train


def load_conversations(path: str | Path) -> list[list[Message]]:
    """Read a JSONL file where each line is {"messages": [{"role", "content"}, ...]}."""
    conversations = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        messages = json.loads(line)["messages"]
        conversations.append([Message(m["role"], m["content"]) for m in messages])
    return conversations


def run_sft(cfg: Config, base_ckpt: str, tokenizer_spec: str, sft_path: str) -> None:
    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    print(f"device: {describe(device)}")

    ckpt = load_checkpoint(base_ckpt, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"]) if ckpt.get("config") else cfg.model
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(model_state(ckpt))
    cfg.model = model_cfg
    print(f"loaded base model from {base_ckpt}")

    tokenizer = get_tokenizer(tokenizer_spec)
    pad_id = tokenizer.encode(PAD)[0]
    conversations = load_conversations(sft_path)
    dataset = SFTDataset(conversations, tokenizer, model_cfg.context_length)
    if len(dataset) == 0:
        raise ValueError(f"no trainable conversations in {sft_path} — every example needs an assistant turn")
    print(f"{len(dataset)} SFT examples")

    loader = DataLoader(
        dataset,
        batch_size=min(cfg.train.batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate_lm, pad_id=pad_id),
    )

    optimizer = build_optimizer(model, cfg.train)
    train(
        model,
        optimizer,
        infinite(loader),
        cfg.train,
        device,
        out_dir=cfg.train.out_dir,
        full_config=cfg,
    )
