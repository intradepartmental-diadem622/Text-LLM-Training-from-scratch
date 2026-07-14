from textllm.tokenizer.bpe import SPLIT_PATTERN, Tokenizer, TiktokenTokenizer

__all__ = [
    "Tokenizer",
    "TiktokenTokenizer",
    "SPLIT_PATTERN",
    "get_tokenizer",
    "special_id",
    "check_vocab_fit",
]


def get_tokenizer(spec: str):
    """Load a tokenizer from a spec string.

    ``"tiktoken:gpt2"`` wraps a pretrained tiktoken encoding; anything else is treated as
    a path to a JSON file saved by our own BPE ``Tokenizer``.
    """
    if spec.startswith("tiktoken:"):
        return TiktokenTokenizer(spec.split(":", 1)[1])
    return Tokenizer.load(spec)


def special_id(tokenizer, token: str) -> int:
    """Id of a registered special token; raises if the tokenizer lacks it."""
    ids = tokenizer.encode(token)
    if len(ids) != 1:
        raise ValueError(
            f"{token!r} is not a special token in this tokenizer; chat training and "
            f"inference need a tokenizer trained with the textllm chat specials"
        )
    return ids[0]


def check_vocab_fit(tokenizer, vocab_size: int) -> None:
    """Reject a tokenizer whose ids the model's embedding cannot index."""
    top = getattr(tokenizer, "max_token_id", None)
    if top is not None and top >= vocab_size:
        raise ValueError(
            f"tokenizer ids reach {top} but the model vocabulary has {vocab_size} "
            f"entries; raise model.vocab_size or use a smaller tokenizer"
        )
