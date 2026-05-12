"""Apply a BufferAction to a ReplayBuffer in place."""

from __future__ import annotations

from hermes_dqn.agent.replay_buffer import ReplayBuffer
from hermes_dqn.buffer.policy import BufferAction


def apply_policy(
    buffer: ReplayBuffer,
    action: BufferAction,
    decay_factor: float = 0.5,
) -> None:
    """Mutate buffer in place per action. KEEP is a no-op. Raises on unknown action."""
    if action is BufferAction.KEEP:
        return
    if action is BufferAction.DECAY:
        buffer.decay_weights(decay_factor)
        return
    if action is BufferAction.CLEAR:
        buffer.clear()
        return
    raise ValueError(f"Unknown BufferAction: {action!r}")
