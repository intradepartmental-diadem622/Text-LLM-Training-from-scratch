import torch

from textllm.config import ModelConfig
from textllm.model import Transformer


def _model(**kw):
    defaults = dict(vocab_size=128, context_length=64, n_embed=64, n_head=4, n_kv_head=2, n_blocks=3)
    defaults.update(kw)
    torch.manual_seed(0)
    return Transformer(ModelConfig(**defaults)).eval()


def test_forward_shapes():
    model = _model()
    idx = torch.randint(0, 128, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 128)
    assert loss.dim() == 0


def test_kv_cache_matches_full_forward():
    model = _model()
    idx = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        full, _ = model(idx)
        caches = model.new_caches()
        steps = [model(idx[:, t : t + 1], caches=caches, start_pos=t)[0] for t in range(idx.shape[1])]
        cached = torch.cat(steps, dim=1)
    assert torch.allclose(full, cached, atol=1e-4)


def test_naive_and_sdpa_agree():
    idx = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        fast, _ = _model(naive_attention=False)(idx)
        slow, _ = _model(naive_attention=True)(idx)
    assert torch.allclose(fast, slow, atol=1e-4)


def test_gqa_reduces_kv_params():
    full = _model(n_kv_head=4)
    grouped = _model(n_kv_head=1)
    assert grouped.num_params() < full.num_params()


def test_weight_tying():
    model = _model(tie_weights=True)
    assert model.lm_head.weight is model.token_embed.weight


def test_chunked_prefill_matches_full_forward():
    # Exercises the decode mask with T > 1 and a non-empty cache — the path a long
    # prompt takes if it's ever fed in pieces.
    model = _model()
    idx = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        full = model.forward_hidden(idx)
        caches = model.new_caches()
        h1 = model.forward_hidden(idx[:, :8], caches, start_pos=0)
        h2 = model.forward_hidden(idx[:, 8:], caches, start_pos=8)
    assert torch.allclose(full, torch.cat([h1, h2], dim=1), atol=1e-4)


def test_rope_is_identity_at_position_zero():
    from textllm.model.rope import RotaryEmbedding, apply_rotary

    rope = RotaryEmbedding(16)
    cos, sin = rope(torch.tensor([0]))
    x = torch.randn(1, 2, 1, 16)
    assert torch.allclose(apply_rotary(x, cos, sin), x, atol=1e-6)


def test_rope_preserves_vector_norm():
    from textllm.model.rope import RotaryEmbedding, apply_rotary

    rope = RotaryEmbedding(16)
    cos, sin = rope(torch.tensor([7]))
    x = torch.randn(1, 2, 1, 16)
    rotated = apply_rotary(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)
