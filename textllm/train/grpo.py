"""GRPO — Group Relative Policy Optimization (the RLVR recipe).

For each prompt we sample a group of completions, score them with a verifiable reward, and
turn the scores into advantages by normalizing within the group (no separate value network
needed — the group average is the baseline). The policy is then pushed toward the
above-average completions, with a KL penalty back to a frozen reference keeping it stable.
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F

from textllm.config import Config, ModelConfig
from textllm.device import describe, pick_device
from textllm.infer.generate import generate
from textllm.model import Transformer
from textllm.tokenizer import check_vocab_fit, get_tokenizer
from textllm.train.loop import build_optimizer, load_checkpoint, model_state, save_checkpoint
from textllm.train.rewards import RewardFn


def _token_logprobs(model, seq: torch.Tensor) -> torch.Tensor:
    """Log-prob of each realized next token; returns shape (B, T-1)."""
    logits, _ = model(seq)
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)


def run_grpo(
    cfg: Config,
    base_ckpt: str,
    tokenizer_spec: str,
    prompts: list[str],
    reward_fn: RewardFn,
    *,
    group_size: int = 6,
    beta: float = 0.02,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
) -> None:
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
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
    pad_id = tokenizer.encode(" ")[0]  # any real token id; padded positions are masked out
    optimizer = build_optimizer(policy, cfg.train)

    for step in range(cfg.train.steps):
        prompt = random.choice(prompts)
        prompt_ids = tokenizer.encode(prompt)[: model_cfg.context_length // 2]

        # Roll out a group of completions from the current policy.
        samples, rewards = [], []
        policy.eval()
        for _ in range(group_size):
            completion = generate(
                policy, prompt_ids, device,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            text = tokenizer.decode(completion)
            samples.append((prompt_ids, completion))
            rewards.append(reward_fn(prompt, text))
        policy.train()

        advantages = _group_advantages(rewards, device)
        seq, resp_mask = _pack(samples, pad_id, device)

        pol_lp = _token_logprobs(policy, seq)
        with torch.no_grad():
            ref_lp = _token_logprobs(reference, seq)

        # Policy-gradient term pushes toward high-advantage tokens; k3 KL keeps us near ref.
        pg = -advantages[:, None] * pol_lp
        kl = torch.exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1.0
        loss = ((pg + beta * kl) * resp_mask).sum() / resp_mask.sum().clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % 10 == 0 or step == cfg.train.steps - 1:
            print(f"step {step:>5} | mean reward {sum(rewards) / len(rewards):.3f} | loss {loss.item():.4f}")

    save_checkpoint(f"{cfg.train.out_dir}/final.pt", policy, optimizer, cfg.train.steps, cfg)


def _group_advantages(rewards: list[float], device) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32, device=device)
    return (r - r.mean()) / (r.std() + 1e-6)


def _pack(samples, pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad (prompt, completion) pairs into a batch and mark the completion tokens."""
    seqs = [p + c for p, c in samples]
    length = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), length - 1), dtype=torch.float32, device=device)
    for i, (prompt, completion) in enumerate(samples):
        full = prompt + completion
        batch[i, : len(full)] = torch.tensor(full, device=device)
        # Position t predicts token t+1; supervise where t+1 falls in the completion span.
        start = len(prompt) - 1
        mask[i, start : len(full) - 1] = 1.0
    return batch, mask
