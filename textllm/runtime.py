"""Rebuild a trained model from a checkpoint for inference or further training."""

from __future__ import annotations


from textllm.config import ModelConfig
from textllm.model import Transformer
from textllm.train.loop import load_checkpoint, model_state


def load_model(ckpt_path: str, device) -> tuple[Transformer, ModelConfig]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if not ckpt.get("config"):
        raise ValueError(f"{ckpt_path} has no saved config; cannot rebuild the model")
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = Transformer(model_cfg).to(device)
    model.load_state_dict(model_state(ckpt))
    model.eval()
    return model, model_cfg
