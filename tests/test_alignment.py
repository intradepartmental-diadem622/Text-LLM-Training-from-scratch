import copy

import torch

from textllm.config import ModelConfig
from textllm.model import Transformer
from textllm.train.dpo import dpo_loss, sequence_logprob
from textllm.train.reward import RewardModel, reward_loss


def _model_cfg():
    return ModelConfig(vocab_size=64, context_length=32, n_embed=64, n_head=4, n_kv_head=2, n_blocks=2)


def test_sft_with_no_trainable_examples_raises_clearly(tmp_path):
    import json

    import pytest

    from textllm.chat_template import SPECIAL_TOKENS
    from textllm.config import Config, TrainConfig
    from textllm.tokenizer import Tokenizer
    from textllm.train.loop import save_checkpoint
    from textllm.train.sft import run_sft

    tok = Tokenizer()
    tok.train("hello world how are you " * 20, vocab_size=300)
    tok.add_special(SPECIAL_TOKENS)
    tok.save(tmp_path / "tok.json")

    cfg = Config(_model_cfg(), TrainConfig(device="cpu"))
    save_checkpoint(tmp_path / "base.pt", Transformer(cfg.model), None, 0, cfg)

    # user-only conversations: prompt masking leaves nothing to supervise
    rows = [{"messages": [{"role": "user", "content": "hello"}]}]
    (tmp_path / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    with pytest.raises(ValueError, match="assistant"):
        run_sft(cfg, str(tmp_path / "base.pt"), str(tmp_path / "tok.json"), str(tmp_path / "sft.jsonl"))


def test_reward_score_ignores_padding():
    torch.manual_seed(0)
    rm = RewardModel(_model_cfg()).eval()
    real = torch.tensor([[5, 6, 7, 8]])
    padded = torch.tensor([[5, 6, 7, 8, 0, 0]])  # 0 is the pad id here
    with torch.no_grad():
        s_real = rm.score(real, pad_id=0)
        s_padded = rm.score(padded, pad_id=0)
    assert torch.allclose(s_real, s_padded, atol=1e-5)


def test_sequence_logprob_counts_only_supervised():
    torch.manual_seed(0)
    model = Transformer(_model_cfg()).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    masked = torch.full_like(ids, -100)
    with torch.no_grad():
        assert sequence_logprob(model, ids, masked).item() == 0.0


def test_dpo_loss_decreases_on_fixed_pair():
    torch.manual_seed(0)
    cfg = _model_cfg()
    policy = Transformer(cfg)
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    batch = {
        "chosen": (torch.tensor([[1, 2, 3, 4]]), torch.tensor([[-100, 2, 3, 4]])),
        "rejected": (torch.tensor([[1, 5, 6, 7]]), torch.tensor([[-100, 5, 6, 7]])),
    }
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    first = dpo_loss(policy, reference, batch, beta=0.1).item()
    for _ in range(50):
        loss = dpo_loss(policy, reference, batch, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_grpo_group_advantages_are_normalized():
    from textllm.train.grpo import _group_advantages

    adv = _group_advantages([1.0, 0.0, 1.0, 0.0], torch.device("cpu"))
    assert abs(adv.mean().item()) < 1e-6
    assert adv[0] > 0 > adv[1]  # rewarded samples sit above the group baseline


def test_grpo_pack_masks_only_completion_tokens():
    from textllm.train.grpo import _pack

    samples = [([1, 2, 3], [4, 5]), ([1], [6, 7, 8])]
    batch, mask = _pack(samples, pad_id=0, device=torch.device("cpu"))
    assert batch.shape == (2, 5) and mask.shape == (2, 4)
    # Row 0: prompt is 3 tokens, so positions 2..3 predict the completion tokens 4, 5.
    assert mask[0].tolist() == [0, 0, 1, 1]
    # Row 1: prompt is 1 token, completion is 3 — positions 0..2 supervised, padding not.
    assert mask[1].tolist() == [1, 1, 1, 0]
    assert batch[1, 4] == 0  # padded with pad_id


def test_reward_loss_separates_pair():
    torch.manual_seed(0)
    rm = RewardModel(_model_cfg())
    batch = {
        "chosen": (torch.tensor([[1, 2, 3, 4]]), None),
        "rejected": (torch.tensor([[1, 5, 6, 7]]), None),
    }
    opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    first = reward_loss(rm, batch, pad_id=0).item()
    for _ in range(50):
        loss = reward_loss(rm, batch, pad_id=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
