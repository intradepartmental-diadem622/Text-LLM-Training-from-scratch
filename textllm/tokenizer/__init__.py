from textllm.tokenizer.bpe import SPLIT_PATTERN, Tokenizer, TiktokenTokenizer

__all__ = ["Tokenizer", "TiktokenTokenizer", "SPLIT_PATTERN", "get_tokenizer"]


def get_tokenizer(spec: str):
    """Load a tokenizer from a spec string.

    ``"tiktoken:gpt2"`` wraps a pretrained tiktoken encoding; anything else is treated as
    a path to a JSON file saved by our own BPE ``Tokenizer``.
    """
    if spec.startswith("tiktoken:"):
        return TiktokenTokenizer(spec.split(":", 1)[1])
    return Tokenizer.load(spec)
