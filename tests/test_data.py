import torch

from textllm.chat_template import Message, encode_supervised
from textllm.data.datasets import collate_lm, collate_preference
from textllm.tokenizer import Tokenizer
from textllm.chat_template import SPECIAL_TOKENS


def _tok():
    tok = Tokenizer()
    tok.train("hello world how are you doing today friend " * 20, vocab_size=350)
    tok.add_special(SPECIAL_TOKENS)
    return tok


def test_sft_masks_prompt_only():
    tok = _tok()
    conv = [Message("user", "hello world"), Message("assistant", "how are you")]
    ids, targets = encode_supervised(tok, conv)
    assert len(ids) == len(targets)
    # Something is supervised (the assistant turn) and something is masked (the prompt).
    assert any(t != -100 for t in targets)
    assert any(t == -100 for t in targets)
    # Every non-masked target equals the next input token.
    for i, t in enumerate(targets):
        if t != -100:
            assert t == ids[i + 1]


def test_collate_lm_pads_with_ignore_index():
    a = (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    b = (torch.tensor([4, 5]), torch.tensor([4, 5]))
    x, y = collate_lm([a, b], pad_id=0)
    assert x.shape == (2, 3) and y.shape == (2, 3)
    assert x[1, 2] == 0        # input padded with pad_id
    assert y[1, 2] == -100     # target padded with the ignore index


def test_collate_preference_shapes():
    ex = {"chosen": ([1, 2, 3], [1, 2, 3]), "rejected": ([4, 5], [4, 5])}
    batch = collate_preference([ex, ex], pad_id=0)
    assert batch["chosen"][0].shape == (2, 3)
    assert batch["rejected"][0].shape == (2, 2)


def test_shards_roundtrip(tmp_path):
    from textllm.chat_template import EOT
    from textllm.data import build_bin, load_bin

    tok = _tok()
    text = "hello world how are you"
    n = build_bin(tok, [text, text], tmp_path / "t.bin", eot=EOT)
    data = load_bin(tmp_path / "t.bin")
    assert len(data) == n
    # The stream decodes back to the two documents joined by the separator.
    assert tok.decode(list(data)) == f"{text}{EOT}{text}{EOT}"


def test_build_bin_rejects_vocab_too_large_for_uint16(tmp_path):
    import pytest

    from textllm.data import build_bin

    class HugeVocab:
        vocab_size = 200_000

        def encode(self, text):
            return [0]

    with pytest.raises(ValueError, match="overflow"):
        build_bin(HugeVocab(), ["hello"], tmp_path / "t.bin")


def test_build_bin_handles_multi_token_separator(tmp_path):
    from textllm.data import build_bin, load_bin

    tok = _tok()
    sep = "===DOC==="  # not a registered special token: encodes to several ids
    sep_ids = tok.encode(sep)
    assert len(sep_ids) > 1
    n = build_bin(tok, ["hello world", "how are you"], tmp_path / "t.bin", eot=sep)
    data = list(load_bin(tmp_path / "t.bin"))
    assert len(data) == n
    # the full separator sequence lands after each document, not just its first id
    assert data[-len(sep_ids):] == sep_ids


def test_missing_bin_error_names_the_fix(tmp_path):
    import pytest

    from textllm.data import load_bin

    with pytest.raises(FileNotFoundError, match="llm data"):
        load_bin(tmp_path / "nope.bin")


def test_pretrain_dataset_windows(tmp_path):
    import numpy as np

    from textllm.data import PretrainDataset

    np.arange(100, dtype=np.uint16).tofile(tmp_path / "t.bin")
    ds = PretrainDataset(tmp_path / "t.bin", block_size=10)
    assert len(ds) == 90
    x, y = ds[0]
    assert x.tolist() == list(range(10))
    assert y.tolist() == list(range(1, 11))  # targets are inputs shifted by one
    ds[len(ds) - 1]  # the last window must not run off the end


def test_preference_pair_dropped_when_prompt_fills_context():
    from textllm.chat_template import Message
    from textllm.data import PreferenceDataset

    tok = _tok()
    long_prompt = [Message("user", "hello world how are you doing today friend " * 30)]
    ds = PreferenceDataset([(long_prompt, "fine", "bad")], tok, block_size=16)
    assert len(ds) == 0  # truncation left no supervised response tokens on either side
