from pathlib import Path

import pytest

from textllm.config import Config, ModelConfig, apply_overrides

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_save_load_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.n_embed = 512
    cfg.train.lr = 1e-4
    path = tmp_path / "cfg.json"
    cfg.save(path)
    assert Config.load(path).to_dict() == cfg.to_dict()


def test_overrides_coerce_to_field_types():
    cfg = Config()
    apply_overrides(cfg, {"train.steps": "123", "train.lr": "1e-4", "model.naive_attention": "true"})
    assert cfg.train.steps == 123 and isinstance(cfg.train.steps, int)
    assert cfg.train.lr == 1e-4 and isinstance(cfg.train.lr, float)
    assert cfg.model.naive_attention is True


def test_unknown_override_raises():
    with pytest.raises(KeyError, match="model.typo"):
        apply_overrides(Config(), {"model.typo": "1"})


def test_gqa_divisibility_enforced():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_head=6, n_kv_head=4)


def test_presets_load():
    for name in ("tiny", "small", "base"):
        cfg = Config.load(CONFIGS / f"{name}.json")
        assert cfg.model.head_dim() > 0
