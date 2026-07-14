"""Direct Preference Optimization.

DPO skips training a separate reward model and the RL loop. It nudges the policy to raise
the log-probability of chosen responses over rejected ones, while a frozen reference copy
keeps it from drifting too far. beta controls how hard it pushes.
"""

from __future__ import annotations

import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textllm.chat_template import PAD
from textllm.config import Config, ModelConfig
from textllm.data import PreferenceDataset, collate_preference
from textllm.device import describe, pick_device
from textllm.model import Transformer
from textllm.tokenizer import check_vocab_fit, get_tokenizer, special_id
from textllm.train.loop import build_optimizer, infinite, load_checkpoint, model_state, train
from textllm.train.reward import load_preferences


def sequence_logprob(model: nn.Module, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sum of log-probabilities of the supervised (target != -100) tokens per sequence.

    The sum follows the original DPO paper; it weights long responses more than a
    length-normalized variant (SimPO-style averaging) would.
    """
    logits, _ = model(ids)
    logp = F.log_softmax(logits, dim=-1)
    mask = targets != -100
    picked = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=1)


def dpo_loss(policy: nn.Module, reference: nn.Module, batch: dict, beta: float) -> torch.Tensor:
    (cx, cy), (rx, ry) = batch["chosen"], batch["rejected"]

    pol_chosen = sequence_logprob(policy, cx, cy)
    pol_rejected = sequence_logprob(policy, rx, ry)
    with torch.no_grad():
        ref_chosen = sequence_logprob(reference, cx, cy)
        ref_rejected = sequence_logprob(reference, rx, ry)

    margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()


def run_dpo(cfg: Config, base_ckpt: str, tokenizer_spec: str, pref_path: str, beta: float = 0.1) -> None:
    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    print(f"device: {describe(device)}")

    ckpt = load_checkpoint(base_ckpt, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"]) if ckpt.get("config") else cfg.model
    cfg.model = model_cfg

    policy = Transformer(model_cfg).to(device)
    policy.load_state_dict(model_state(ckpt))
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    tokenizer = get_tokenizer(tokenizer_spec)
    check_vocab_fit(tokenizer, model_cfg.vocab_size)
    pad_id = special_id(tokenizer, PAD)
    dataset = PreferenceDataset(load_preferences(pref_path), tokenizer, model_cfg.context_length)
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
    optimizer = build_optimizer(policy, cfg.train)
    train(
        policy,
        optimizer,
        infinite(loader),
        cfg.train,
        device,
        loss_fn=lambda m, b: dpo_loss(m, reference, b, beta),
        out_dir=cfg.train.out_dir,
        full_config=cfg,
    )
