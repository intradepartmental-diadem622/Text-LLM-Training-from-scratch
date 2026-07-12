import torch

from textllm.config import Config, ModelConfig, TrainConfig
from textllm.model import Transformer
from textllm.runtime import load_model
from textllm.train.loop import build_optimizer, save_checkpoint


def test_save_and_reload_reproduces_outputs(tmp_path):
    cfg = Config(
        ModelConfig(vocab_size=64, context_length=32, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2),
        TrainConfig(),
    )
    torch.manual_seed(0)
    model = Transformer(cfg.model)
    opt = build_optimizer(model, cfg.train)

    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        before, _ = model(idx)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=5, full_config=cfg)

    reloaded, _ = load_model(str(path), torch.device("cpu"))
    with torch.no_grad():
        after, _ = reloaded(idx)
    assert torch.allclose(before, after, atol=1e-6)


def test_compiled_checkpoint_prefix_is_stripped(tmp_path):
    # torch.compile saves weights under "_orig_mod." — every loader must accept them.
    import dataclasses

    from textllm.train.loop import model_state

    cfg = Config(
        ModelConfig(vocab_size=64, context_length=32, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2),
        TrainConfig(),
    )
    torch.manual_seed(0)
    model = Transformer(cfg.model)
    prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}

    ckpt = {"model": prefixed, "optimizer": None, "step": 0, "config": dataclasses.asdict(cfg)}
    assert set(model_state(ckpt)) == set(model.state_dict())

    torch.save(ckpt, tmp_path / "compiled.pt")
    from textllm.runtime import load_model

    reloaded, _ = load_model(str(tmp_path / "compiled.pt"), torch.device("cpu"))
    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = reloaded(idx)
    assert torch.allclose(a, b, atol=1e-6)


def test_perplexity_rejects_a_file_too_short_to_evaluate(tmp_path):
    import dataclasses

    import numpy as np
    import pytest

    from textllm.eval import perplexity

    cfg = Config(
        ModelConfig(vocab_size=64, context_length=32, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2),
        TrainConfig(),
    )
    model = Transformer(cfg.model)
    ckpt = {"model": model.state_dict(), "optimizer": None, "step": 0, "config": dataclasses.asdict(cfg)}
    torch.save(ckpt, tmp_path / "m.pt")
    np.arange(8, dtype=np.uint16).tofile(tmp_path / "tiny.bin")  # < context_length + 1

    with pytest.raises(ValueError, match="too short"):
        perplexity(str(tmp_path / "m.pt"), str(tmp_path / "tiny.bin"))
