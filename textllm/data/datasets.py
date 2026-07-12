"""Datasets for each training stage, plus the collate functions that pad a batch.

Pretraining reads fixed-length windows straight from a memory-mapped token file. SFT and
preference data are variable length, so their collates pad inputs with a pad token and
targets with -100 (the ignore index) so padding never contributes to the loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from textllm.chat_template import Message, encode_supervised
from textllm.data.shards import load_bin


class PretrainDataset(Dataset):
    """Sliding windows over one big token stream; item i predicts token i+1 onward."""

    def __init__(self, bin_path: str, block_size: int) -> None:
        self.data = load_bin(bin_path)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = np.asarray(self.data[i : i + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


class SFTDataset(Dataset):
    """Instruction/response conversations with the prompt masked out of the loss."""

    def __init__(self, conversations: list[list[Message]], tokenizer, block_size: int) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        for conv in conversations:
            ids, targets = encode_supervised(tokenizer, conv)
            ids, targets = ids[:block_size], targets[:block_size]
            if any(t != -100 for t in targets):  # skip examples with nothing to learn
                self.examples.append((ids, targets))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, targets = self.examples[i]
        return torch.tensor(ids), torch.tensor(targets)


class PreferenceDataset(Dataset):
    """Chosen/rejected response pairs for reward modeling and DPO.

    Each item carries both responses already encoded against the same prompt, with a mask
    (targets != -100) marking the response tokens whose log-probabilities the losses sum.
    """

    def __init__(self, items: list[tuple[list[Message], str, str]], tokenizer, block_size: int) -> None:
        self.examples = []
        for prompt, chosen, rejected in items:
            c = encode_supervised(tokenizer, [*prompt, Message("assistant", chosen)])
            r = encode_supervised(tokenizer, [*prompt, Message("assistant", rejected)])
            chosen_pair = (c[0][:block_size], c[1][:block_size])
            rejected_pair = (r[0][:block_size], r[1][:block_size])
            # drop pairs where truncation left no response tokens to supervise
            if not any(t != -100 for t in chosen_pair[1]):
                continue
            if not any(t != -100 for t in rejected_pair[1]):
                continue
            self.examples.append({"chosen": chosen_pair, "rejected": rejected_pair})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


def collate_lm(batch, pad_id: int):
    """Pad a list of (input_ids, targets) to the longest in the batch."""
    xs, ys = zip(*batch)
    x = _pad(xs, pad_id)
    y = _pad(ys, -100)
    return x, y


def collate_preference(batch, pad_id: int):
    def side(key):
        xs = [torch.tensor(ex[key][0]) for ex in batch]
        ys = [torch.tensor(ex[key][1]) for ex in batch]
        return _pad(xs, pad_id), _pad(ys, -100)

    cx, cy = side("chosen")
    rx, ry = side("rejected")
    return {"chosen": (cx, cy), "rejected": (rx, ry)}


def _pad(seqs, pad_value: int) -> torch.Tensor:
    length = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), length), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out
