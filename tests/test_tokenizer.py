from textllm.chat_template import SPECIAL_TOKENS
from textllm.tokenizer import Tokenizer


def _trained(tmp_path):
    text = ("the quick brown fox jumps over the lazy dog. " * 40) + "café über 12345!"
    tok = Tokenizer()
    tok.train(text, vocab_size=400)
    tok.add_special(SPECIAL_TOKENS)
    return tok


def test_roundtrip_ascii_and_unicode(tmp_path):
    tok = _trained(tmp_path)
    for s in ["the quick brown fox", "café über", "12345!", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_special_tokens_are_single_ids(tmp_path):
    tok = _trained(tmp_path)
    for special in SPECIAL_TOKENS:
        ids = tok.encode(special)
        assert len(ids) == 1
        assert tok.decode(ids) == special


def test_save_load_preserves_encoding(tmp_path):
    tok = _trained(tmp_path)
    path = tmp_path / "tok.json"
    tok.save(path)
    reloaded = Tokenizer.load(path)
    s = "the quick café <|end|>"
    assert reloaded.encode(s) == tok.encode(s)
    assert reloaded.decode(reloaded.encode(s)) == s


def test_load_restores_saved_special_ids_exactly(tmp_path):
    import json

    tok = _trained(tmp_path)
    path = tmp_path / "tok.json"
    tok.save(path)

    # perturb the stored ids so they are no longer the contiguous re-derivable block
    data = json.loads(path.read_text())
    data["special"] = {t: i + 100 for t, i in data["special"].items()}
    path.write_text(json.dumps(data))

    reloaded = Tokenizer.load(path)
    assert reloaded.special == data["special"]
    for text, i in data["special"].items():
        assert reloaded.encode(text) == [i]


def test_add_special_across_calls_never_collides(tmp_path):
    tok = _trained(tmp_path)
    tok.add_special(["<|tool|>"])
    tok.add_special(["<|result|>"])
    ids = list(tok.special.values())
    assert len(ids) == len(set(ids))


def test_add_special_after_load_does_not_collide(tmp_path):
    tok = _trained(tmp_path)
    path = tmp_path / "tok.json"
    tok.save(path)

    reloaded = Tokenizer.load(path)
    reloaded.add_special(["<|tool|>"])
    ids = list(reloaded.special.values())
    assert len(ids) == len(set(ids))
    assert reloaded.special["<|tool|>"] > max(tok.special.values())


def test_train_after_add_special_raises(tmp_path):
    import pytest

    tok = Tokenizer()
    tok.add_special(["<|end|>"])
    with pytest.raises(ValueError, match="after training"):
        tok.train("some text " * 20, vocab_size=300)


def test_load_rejects_colliding_special_ids(tmp_path):
    import json

    import pytest

    tok = _trained(tmp_path)
    path = tmp_path / "tok.json"
    tok.save(path)

    data = json.loads(path.read_text())
    first = next(iter(data["special"]))
    data["special"][first] = 260  # a merge id
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="special token ids"):
        Tokenizer.load(path)


def test_longer_special_token_wins_over_its_prefix(tmp_path):
    tok = _trained(tmp_path)
    tok.add_special(["<|t|>", "<|t|>x"])
    assert tok.encode("<|t|>x") == [tok.special["<|t|>x"]]


def test_special_id_and_vocab_fit_helpers(tmp_path):
    import pytest

    from textllm.tokenizer import check_vocab_fit, special_id

    tok = _trained(tmp_path)
    assert special_id(tok, "<|pad|>") == tok.special["<|pad|>"]
    check_vocab_fit(tok, tok.max_token_id + 1)
    with pytest.raises(ValueError, match="vocabulary"):
        check_vocab_fit(tok, tok.max_token_id)

    bare = Tokenizer()
    bare.train("some text here " * 20, vocab_size=300)
    with pytest.raises(ValueError, match="special"):
        special_id(bare, "<|pad|>")
