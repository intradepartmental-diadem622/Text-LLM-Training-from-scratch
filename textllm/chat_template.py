"""How conversations are turned into a single token stream.

A small ChatML-style format with explicit role markers. The same rendering is used at
training time (so SFT masks everything except the assistant's reply) and at chat time
(so we can stop generating when the model emits the end marker).
"""

from __future__ import annotations

from dataclasses import dataclass

BOS = "<|begin|>"
EOT = "<|end|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
PAD = "<|pad|>"

SPECIAL_TOKENS = [BOS, EOT, SYSTEM, USER, ASSISTANT, PAD]


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str


_ROLE_MARKER = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def render(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """Render a conversation to text.

    With ``add_generation_prompt`` the string ends right after the assistant marker, which
    is how you ask the model to start replying at inference time.
    """
    parts = [BOS]
    for msg in messages:
        marker = _ROLE_MARKER[msg.role]
        parts.append(f"{marker}\n{msg.content}{EOT}\n")
    if add_generation_prompt:
        parts.append(f"{ASSISTANT}\n")
    return "".join(parts)


def encode_supervised(tokenizer, messages: list[Message]) -> tuple[list[int], list[int]]:
    """Encode a conversation into (input_ids, target_ids) for SFT.

    Target ids mirror input ids shifted by one, but every token that isn't part of an
    assistant turn is set to -100 so the loss ignores it — the model is only trained to
    produce the assistant's words, not to parrot the prompt.
    """
    input_ids: list[int] = tokenizer.encode(BOS)
    supervised: list[bool] = [False] * len(input_ids)

    for msg in messages:
        marker = _ROLE_MARKER[msg.role]
        header = tokenizer.encode(f"{marker}\n")
        body = tokenizer.encode(f"{msg.content}{EOT}\n")
        input_ids += header + body
        supervised += [False] * len(header)
        supervised += [msg.role == "assistant"] * len(body)

    targets = [-100] * len(input_ids)
    for i in range(len(input_ids) - 1):
        if supervised[i + 1]:
            targets[i] = input_ids[i + 1]
    return input_ids, targets
