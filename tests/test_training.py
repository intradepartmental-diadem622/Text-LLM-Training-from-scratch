import pytest
import torch
from torch.utils.data import DataLoader

from textllm.config import TrainConfig
from textllm.train.loop import infinite, lr_at


def test_lr_warmup_and_decay():
    cfg = TrainConfig(steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-4)
    assert lr_at(0, cfg) < cfg.lr                 # warmup starts below peak
    assert abs(lr_at(99, cfg) - cfg.lr) < 1e-9    # reaches peak at end of warmup
    mid = lr_at(550, cfg)
    assert cfg.min_lr < mid < cfg.lr              # decaying through the middle
    assert abs(lr_at(1000, cfg) - cfg.min_lr) < 1e-9


def test_infinite_raises_instead_of_hanging_on_empty_loader():
    loader = DataLoader([], batch_size=4, drop_last=True)
    with pytest.raises(ValueError, match="batch_size"):
        next(infinite(loader))


def test_overfits_tiny_batch():
    # A model with enough capacity should drive the loss on one fixed batch toward zero.
    from textllm.config import ModelConfig
    from textllm.model import Transformer
    from textllm.train.loop import build_optimizer

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, context_length=32, n_embed=64, n_head=4, n_kv_head=4, n_blocks=2)
    model = Transformer(cfg)
    opt = build_optimizer(model, TrainConfig(lr=1e-3, weight_decay=0.0))

    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.1
