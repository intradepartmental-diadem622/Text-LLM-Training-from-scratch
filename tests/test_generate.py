import torch

from textllm.config import ModelConfig
from textllm.infer.generate import generate
from textllm.infer.sample import sample_next
from textllm.model import Transformer


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, context_length=32, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2)
    return Transformer(cfg).eval()


def test_greedy_is_deterministic():
    model = _model()
    device = torch.device("cpu")
    a = generate(model, [1, 2, 3], device, max_new_tokens=10, temperature=0)
    b = generate(model, [1, 2, 3], device, max_new_tokens=10, temperature=0)
    assert a == b
    assert len(a) == 10


def test_stop_id_halts_generation():
    model = _model()
    device = torch.device("cpu")
    out = generate(model, [1, 2, 3], device, max_new_tokens=10, temperature=0, stop_ids={_first_greedy(model)})
    assert out == []  # stops immediately when the first token is a stop token


def _first_greedy(model):
    with torch.no_grad():
        logits, _ = model(torch.tensor([[1, 2, 3]]), caches=model.new_caches(), start_pos=0)
    return int(logits[0, -1].argmax())


def test_top_k_restricts_support():
    logits = torch.tensor([0.0, 5.0, 0.0, 4.0, 0.0])
    picks = {sample_next(logits, temperature=1.0, top_k=2) for _ in range(50)}
    assert picks <= {1, 3}  # only the two largest logits can be sampled


def test_top_p_keeps_only_the_dominant_token():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])  # ~99.99% of the mass on token 0
    picks = {sample_next(logits, temperature=1.0, top_p=0.5) for _ in range(50)}
    assert picks == {0}


def test_repetition_penalty_demotes_seen_tokens():
    logits = torch.tensor([3.0, 3.1])  # token 1 barely wins unpenalized
    assert sample_next(logits, temperature=0) == 1
    assert sample_next(logits, temperature=0, repetition_penalty=2.0, prev_ids=[1]) == 0


def test_empty_prompt_is_rejected():
    import pytest

    model = _model()
    with pytest.raises(ValueError, match="empty"):
        generate(model, [], torch.device("cpu"), max_new_tokens=4)


def test_bad_sampling_values_raise_clear_errors():
    import pytest

    logits = torch.zeros(8)
    with pytest.raises(ValueError, match="temperature"):
        sample_next(logits, temperature=-0.5)
    with pytest.raises(ValueError, match="top_k"):
        sample_next(logits, top_k=0)
    with pytest.raises(ValueError, match="top_p"):
        sample_next(logits, top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        sample_next(logits, top_p=1.5)
    with pytest.raises(ValueError, match="repetition_penalty"):
        sample_next(logits, repetition_penalty=0)
