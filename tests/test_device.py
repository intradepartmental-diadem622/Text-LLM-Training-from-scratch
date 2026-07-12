import contextlib

import torch

from textllm.device import autocast, describe, needs_grad_scaler, pick_device


def test_explicit_device_wins():
    assert pick_device("cpu").type == "cpu"


def test_auto_pick_returns_something_usable():
    device = pick_device()
    torch.zeros(1, device=device)  # must be constructible on this machine


def test_cpu_autocast_is_a_noop():
    ctx = autocast(torch.device("cpu"))
    assert isinstance(ctx, contextlib.nullcontext)
    assert not needs_grad_scaler(torch.device("cpu"))


def test_describe_mentions_the_backend():
    device = pick_device()
    assert device.type in describe(device)
