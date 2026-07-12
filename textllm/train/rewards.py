"""Verifiable reward functions for GRPO/RLVR.

A reward function takes the prompt text and a sampled completion and returns a float. These
are deliberately simple and checkable — the whole point of RLVR is that the reward is a
program, not a learned model.
"""

from __future__ import annotations

import re
from typing import Callable

RewardFn = Callable[[str, str], float]


def contains(target: str) -> RewardFn:
    """1.0 if the completion contains ``target``, else 0.0."""
    return lambda prompt, completion: 1.0 if target in completion else 0.0


def exact_answer(answer: str) -> RewardFn:
    """1.0 when the completion's final number matches ``answer``, with a format bonus."""

    def fn(prompt: str, completion: str) -> float:
        reward = 0.0
        numbers = re.findall(r"-?\d+", completion)
        if numbers and numbers[-1] == answer:
            reward += 1.0
        if "the answer is" in completion.lower():  # nudge toward a clear final form
            reward += 0.1
        return reward

    return fn


def length_target(ideal: int, tolerance: int = 10) -> RewardFn:
    """Peaks at ``ideal`` characters and falls off linearly — a toy but non-trivial signal."""
    return lambda prompt, completion: max(0.0, 1.0 - abs(len(completion) - ideal) / max(1, tolerance))
