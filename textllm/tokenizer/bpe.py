"""A byte-level BPE tokenizer, trained from scratch.

Same idea as GPT-2/tiktoken: work on raw UTF-8 bytes so nothing is ever out of
vocabulary, and learn a table of merges that greedily glue together the most frequent
adjacent pairs. Text is first split on a regex so merges never cross word or whitespace
boundaries in ways that hurt compression.

This is the readable, hackable version. If you'd rather not train one, ``from_tiktoken``
wraps an existing tiktoken encoding behind the same interface.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Splits text into word-ish chunks before merging. Contractions, then runs of letters,
# digits, punctuation, and whitespace. \w and \d are Unicode-aware in Python 3, so this
# needs no third-party regex module.
SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, pattern: str = SPLIT_PATTERN) -> None:
        self._re = re.compile(pattern)
        self.pattern = pattern
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}
        self._special_re: re.Pattern | None = None

    # -- training -----------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")
        if self.special:
            # merge ids are assigned from 256 upward and would collide with them
            raise ValueError("add special tokens after training, not before")
        n_merges = vocab_size - 256

        # Count how often each pre-split chunk appears, then only ever work on the unique
        # chunks weighted by their count. Far cheaper than rescanning the whole corpus.
        chunk_counts = Counter(self._re.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        weights = list(chunk_counts.values())

        for step in range(n_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for ids, w in zip(words, weights):
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += w
            if not pair_counts:
                break

            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + step
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = [_merge(ids, best, new_id) for ids in words]

            if verbose and (step + 1) % 100 == 0:
                print(f"merge {step + 1}/{n_merges}: {best} -> {new_id}")

    def add_special(self, tokens: list[str]) -> None:
        """Register special tokens (chat markers, end-of-text) above the learned vocab."""
        next_id = max(self.vocab) + 1 if self.vocab else 256
        if self.special:
            # existing specials live above the vocab too; never hand out their ids again
            next_id = max(next_id, max(self.special.values()) + 1)
        for tok in tokens:
            if tok in self.special:
                continue
            self.special[tok] = next_id
            next_id += 1
        self._compile_special_pattern()

    def _compile_special_pattern(self) -> None:
        if self.special:
            # longest first, so one special token can't shadow a longer one in the regex
            joined = "|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True))
            self._special_re = re.compile(f"({joined})")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    @property
    def max_token_id(self) -> int:
        top = max(self.vocab)
        if self.special:
            top = max(top, max(self.special.values()))
        return top

    # -- encode / decode ----------------------------------------------------------

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        # Repeatedly apply the earliest-learned merge that's still present. Earliest wins
        # because a later merge may depend on an earlier one having happened first.
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as ordinary text."""
        ids: list[int] = []
        for chunk in self._re.findall(text):
            ids.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text, expanding registered special tokens into their ids."""
        if not allowed_special or not self._special_re:
            return self.encode_ordinary(text)

        ids: list[int] = []
        for part in self._special_re.split(text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        # ids past the trained vocab (a model's padded output layer) are skipped
        special_inv = {i: t for t, i in self.special.items()}
        pieces: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                pieces.append(self.vocab[i])
            elif i in special_inv:
                pieces.append(special_inv[i].encode("utf-8"))
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "pattern": self.pattern,
            # JSON keys must be strings, so merges are stored as "a,b": new_id
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "special": self.special,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(pattern=data["pattern"])
        for key, new_id in data["merges"].items():
            a, b = (int(x) for x in key.split(","))
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        # a trained model depends on these exact ids, so restore them verbatim
        tok.special = {t: int(i) for t, i in data.get("special", {}).items()}
        ids = list(tok.special.values())
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids) or set(ids) & set(tok.vocab):
            raise ValueError(
                f"invalid special token ids in {path}: they must be unique, non-negative, "
                f"and outside the merge vocabulary"
            )
        tok._compile_special_pattern()
        return tok

    # -- alternative: reuse a tiktoken encoding -----------------------------------

    @staticmethod
    def from_tiktoken(name: str = "gpt2") -> "TiktokenTokenizer":
        return TiktokenTokenizer(name)


class TiktokenTokenizer:
    """Thin adapter so a pretrained tiktoken encoding fits the same call sites."""

    def __init__(self, name: str = "gpt2") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(name)
        self.special: dict[str, int] = dict(self._enc._special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def max_token_id(self) -> int:
        return self._enc.n_vocab - 1

    def encode_ordinary(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if not allowed_special:
            return self._enc.encode_ordinary(text)  # special strings become plain text
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
