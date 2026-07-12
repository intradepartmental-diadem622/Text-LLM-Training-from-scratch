"""Turn raw text into a flat file of token ids we can memory-map during training.

We store ids as ``uint16`` (fine for any vocab up to 65535, which covers our own BPE and
GPT-2's tiktoken) in a plain ``.bin`` file. Memory-mapping it means training reads windows
straight from disk without loading the whole corpus into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

DTYPE = np.uint16


def build_bin(tokenizer, texts: Iterable[str], out_path: str | Path, eot: str | None = None) -> int:
    """Tokenize ``texts`` into one ``.bin`` file, optionally separating docs with ``eot``.

    Returns the number of tokens written. Documents are flushed as they're encoded so a
    large corpus never has to sit in memory all at once.
    """
    max_id = np.iinfo(DTYPE).max
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None and vocab_size > max_id + 1:
        raise ValueError(
            f"tokenizer vocab ({vocab_size}) exceeds what {np.dtype(DTYPE).name} can store "
            f"({max_id + 1} ids) — token ids would silently overflow"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eot_ids = tokenizer.encode(eot) if eot else []  # a separator can be several tokens

    written = 0
    with open(out_path, "wb") as f:
        for text in texts:
            ids = tokenizer.encode(text)
            ids.extend(eot_ids)
            np.asarray(ids, dtype=DTYPE).tofile(f)
            written += len(ids)
    return written


def load_bin(path: str | Path) -> np.ndarray:
    """Memory-map a token ``.bin`` file as a read-only 1-D array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no token file at {path} — run `llm data` first")
    return np.memmap(path, dtype=DTYPE, mode="r")
