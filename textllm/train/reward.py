"""A Bradley-Terry reward model.

We reuse the transformer backbone and add a single scalar head. The score of a sequence is
read off the last real token. Training maximizes the gap between the chosen and rejected
responses: loss = -log sigmoid(r_chosen - r_rejected).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textllm.chat_template import PAD
from textllm.config import Config, ModelConfig
from textllm.data import PreferenceDataset, collate_preference
from textllm.device import describe, pick_device
from textllm.model import Transformer
from textllm.tokenizer import get_tokenizer
from textllm.train.loop import build_optimizer, infinite, load_checkpoint, model_state, train


class RewardModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = Transformer(cfg)
        self.head = nn.Linear(cfg.n_embed, 1, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.forward_hidden(ids)
        return self.head(hidden).squeeze(-1)  # per-token reward, shape (B, T)

    def score(self, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        rewards = self(ids)
        last = (ids != pad_id).sum(dim=1) - 1  # index of the final real token
        return rewards[torch.arange(ids.shape[0], device=ids.device), last]


def reward_loss(model: RewardModel, batch: dict, pad_id: int) -> torch.Tensor:
    chosen_ids = batch["chosen"][0]
    rejected_ids = batch["rejected"][0]
    r_chosen = model.score(chosen_ids, pad_id)
    r_rejected = model.score(rejected_ids, pad_id)
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def run_reward(cfg: Config, base_ckpt: str, tokenizer_spec: str, pref_path: str) -> None:
    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    print(f"device: {describe(device)}")

    ckpt = load_checkpoint(base_ckpt, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"]) if ckpt.get("config") else cfg.model
    model = RewardModel(model_cfg).to(device)
    model.backbone.load_state_dict(model_state(ckpt))
    cfg.model = model_cfg

    tokenizer = get_tokenizer(tokenizer_spec)
    pad_id = tokenizer.encode(PAD)[0]
    items = load_preferences(pref_path)
    dataset = PreferenceDataset(items, tokenizer, model_cfg.context_length)
    if len(dataset) == 0:
        raise ValueError(f"no preference pairs in {pref_path}")
    print(f"{len(dataset)} preference pairs")

    loader = DataLoader(
        dataset,
        batch_size=min(cfg.train.batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate_preference, pad_id=pad_id),
    )
    optimizer = build_optimizer(model, cfg.train)
    train(
        model,
        optimizer,
        infinite(loader),
        cfg.train,
        device,
        loss_fn=partial(reward_loss, pad_id=pad_id),
        out_dir=cfg.train.out_dir,
        full_config=cfg,
    )


def load_preferences(path: str | Path):
    """Read JSONL of {"prompt": [...messages], "chosen": str, "rejected": str}."""
    import json

    from textllm.chat_template import Message

    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompt = [Message(m["role"], m["content"]) for m in row["prompt"]]
        items.append((prompt, row["chosen"], row["rejected"]))
    return items
