"""Generation accuracy on a small question/answer set.

Reads JSONL of {"question": ..., "answer": ...}, lets the model answer each question in
the chat format, and checks whether the final number matches. This is the same style of
verifiable check GRPO trains against, reused here to measure progress.
"""

from __future__ import annotations

import json
from pathlib import Path

from textllm.chat_template import EOT, Message, render
from textllm.device import pick_device
from textllm.infer.generate import generate
from textllm.runtime import load_model
from textllm.tokenizer import get_tokenizer
from textllm.train.rewards import exact_answer


def accuracy(ckpt_path: str, tokenizer_spec: str, qa_path: str, max_new_tokens: int = 64) -> float:
    device = pick_device()
    model, cfg = load_model(ckpt_path, device)
    tokenizer = get_tokenizer(tokenizer_spec)
    eot_ids = tokenizer.encode(EOT)
    stops = set(eot_ids) if len(eot_ids) == 1 else set()

    rows = [json.loads(line) for line in Path(qa_path).read_text().splitlines() if line.strip()]
    correct = 0
    for row in rows:
        prompt = render([Message("user", row["question"])], add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt)[-cfg.context_length :]
        ids = generate(model, prompt_ids, device, max_new_tokens=max_new_tokens, temperature=0, stop_ids=stops)
        answer = tokenizer.decode(ids)
        if exact_answer(str(row["answer"]))(row["question"], answer) >= 1.0:
            correct += 1

    return correct / max(1, len(rows))
