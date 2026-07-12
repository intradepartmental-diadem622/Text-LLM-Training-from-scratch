"""The one place that knows about hardware.

Everything else asks this module which device to run on and how to autocast.
No other file should reference ``torch.cuda`` or ``.cuda()`` directly, so the whole
project stays honest about running on CPU, Apple Silicon (MPS), and NVIDIA GPUs.
"""

from __future__ import annotations

import contextlib

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or the one named in ``prefer``.

    Order of preference is CUDA, then MPS, then CPU. Passing an explicit name lets a
    caller pin a device (useful for tests that must stay on CPU).
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """The dtype to autocast to, or None when mixed precision won't help.

    bf16 needs Ampere or newer; older CUDA cards fall back to fp16 (which then needs a
    GradScaler during training). MPS and CPU train in fp32 here — autocast on those
    backends is either unsupported or not worth the numerical trouble for a small model.
    """
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def needs_grad_scaler(device: torch.device) -> bool:
    """fp16 training under/overflows without loss scaling; bf16 and fp32 don't."""
    return autocast_dtype(device) is torch.float16


def autocast(device: torch.device):
    """Autocast context for the device, or a no-op when mixed precision is off."""
    dtype = autocast_dtype(device)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def describe(device: torch.device) -> str:
    """Human-readable one-liner for logs."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return f"cuda ({name}, autocast {dtype})"
    if device.type == "mps":
        return "mps (Apple Silicon, fp32)"
    return "cpu (fp32)"
