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
