"""Interactive chat REPL and single-prompt completion over a trained model."""

from __future__ import annotations

import sys

from textllm.chat_template import EOT, Message, render
from textllm.device import pick_device
from textllm.infer.generate import stream
from textllm.runtime import load_model
from textllm.tokenizer import get_tokenizer


def _stop_ids(tokenizer) -> set[int]:
    stops = set()
    for marker in (EOT,):
        try:
            stops.update(tokenizer.encode(marker))
        except Exception:
            pass
    return stops


def chat_repl(ckpt_path: str, tokenizer_spec: str, system: str | None = None, **sampling) -> None:
    device = pick_device()
    model, cfg = load_model(ckpt_path, device)
    tokenizer = get_tokenizer(tokenizer_spec)
    stops = _stop_ids(tokenizer)

    history: list[Message] = []
    if system:
        history.append(Message("system", system))
    print("chat ready — type 'exit' to quit\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ("exit", "quit"):
            break
        if not user:
            continue

        history.append(Message("user", user))
        prompt_ids = tokenizer.encode(render(history, add_generation_prompt=True))
        prompt_ids = prompt_ids[-cfg.context_length :]

        print("bot> ", end="", flush=True)
        pieces: list[int] = []
        for tok in stream(model, prompt_ids, device, max_new_tokens=256, stop_ids=stops, **sampling):
            pieces.append(tok)
            sys.stdout.write(tokenizer.decode([tok]))
            sys.stdout.flush()
        print("\n")
        history.append(Message("assistant", tokenizer.decode(pieces)))


def complete(ckpt_path: str, tokenizer_spec: str, prompt: str, max_new_tokens: int, **sampling) -> str:
    """Raw (non-chat) completion, useful right after pretraining."""
    device = pick_device()
    model, cfg = load_model(ckpt_path, device)
    tokenizer = get_tokenizer(tokenizer_spec)
    prompt_ids = tokenizer.encode(prompt)[-cfg.context_length :]
    out = []
    for tok in stream(model, prompt_ids, device, max_new_tokens=max_new_tokens, **sampling):
        out.append(tok)
    return prompt + tokenizer.decode(out)
