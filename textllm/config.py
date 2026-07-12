"""Typed configuration for the model and each training stage.

Configs are plain dataclasses so they're easy to read and default sensibly. They can be
loaded from JSON, dumped back to JSON (checkpoints carry their model config so inference
never has to guess the shape), and overridden field-by-field from the command line.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_embed: int = 384
    n_head: int = 6
    n_kv_head: int = 6          # set < n_head for grouped-query attention
    n_blocks: int = 6
    mlp_ratio: float = 8 / 3    # SwiGLU keeps ~2/3 of the usual 4x to match param count
    rope_theta: float = 10000.0
    tie_weights: bool = True
    naive_attention: bool = False   # use the readable teaching path instead of SDPA

    def head_dim(self) -> int:
        if self.n_embed % self.n_head != 0:
            raise ValueError("n_embed must be divisible by n_head")
        return self.n_embed // self.n_head

    def __post_init__(self) -> None:
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head for GQA")


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data"
    out_dir: str = "checkpoints"
    # optimization
    batch_size: int = 32
    grad_accum: int = 1
    steps: int = 5000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # bookkeeping
    eval_every: int = 250
    eval_iters: int = 100
    save_every: int = 0         # 0 => only save at the end
    keep_last: int = 3
    seed: int = 1337
    device: str | None = None   # None => auto-pick
    compile: bool = False       # torch.compile the model (CUDA/CPU; skipped on MPS)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        model = ModelConfig(**data.get("model", {}))
        train = TrainConfig(**data.get("train", {}))
        return cls(model=model, train=train)


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply ``section.field=value`` overrides collected from the CLI.

    Keys look like ``model.n_embed`` or ``train.lr``; values are coerced to the field's
    declared type so ``--set train.lr=1e-4`` lands as a float, not a string.
    """
    for dotted, raw in overrides.items():
        section, _, field_name = dotted.partition(".")
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, field_name):
            raise KeyError(f"unknown config field: {dotted}")
        current = getattr(target, field_name)
        setattr(target, field_name, _coerce(raw, type(current)))
    return cfg


def _coerce(raw: Any, to_type: type) -> Any:
    if to_type is bool:
        return str(raw).lower() in ("1", "true", "yes", "on")
    if to_type in (int, float, str):
        return to_type(raw)
    return raw  # None-typed or unknown fields pass through untouched
