"""Replay-buffer policy decision from a RewardDiff."""

from __future__ import annotations

from enum import Enum

from hermes_dqn.buffer.ast_diff import RewardDiff


class BufferAction(Enum):
    KEEP = "keep"
    DECAY = "decay"
    CLEAR = "clear"


_DIFF_TO_ACTION: dict[str, BufferAction] = {
    "IDENTICAL": BufferAction.KEEP,
    "NUMERIC_DIFF": BufferAction.KEEP,
    "STRUCTURAL_DIFF": BufferAction.DECAY,
    "TOTAL_REWRITE": BufferAction.CLEAR,
}


def decide_policy(diff: RewardDiff) -> BufferAction:
    """Map a RewardDiff kind to the recommended BufferAction."""
    try:
        return _DIFF_TO_ACTION[diff.kind]
    except KeyError as e:
        raise ValueError(f"Unknown RewardDiff.kind: {diff.kind!r}") from e
