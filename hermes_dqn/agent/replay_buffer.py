"""Uniform replay buffer backed by numpy circular arrays.

Supports optional per-sample weights for use by `ast-buffer-manager`'s
DECAY policy: when reward changes between training iterations, older
experience can have its sampling probability scaled down without
emptying the buffer. When all weights equal 1.0, `sample()` takes a
deterministic fast path byte-identical to the original uniform sampler
so `bootstrap-dqn-baseline`'s reproducibility guarantee is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer with optional per-sample sampling weights."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._weights = np.ones((capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._dones[i] = float(done)
        # New samples always start at full weight; overwrites also reset the slot.
        self._weights[i] = 1.0
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        size = self._size
        weights = self._weights[:size]
        # Fast path: identical to the original uniform sampler when all weights
        # are 1.0. Preserves byte-deterministic sample sequences for any caller
        # that never invokes decay_weights — i.e., bootstrap-dqn-baseline and
        # gemma-reward-generator era runs.
        if bool(np.all(weights == 1.0)):
            idx = self._rng.integers(0, size, size=batch_size)
        else:
            probs = weights / weights.sum()
            idx = self._rng.choice(size, size=batch_size, p=probs, replace=True)
        return Batch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            dones=self._dones[idx],
        )

    def decay_weights(self, factor: float) -> None:
        """Scale all currently-stored samples' weights by `factor`."""
        # Only existing samples; new pushes still arrive at weight 1.0 (see push()).
        self._weights[: self._size] *= float(factor)

    def clear(self) -> None:
        """Reset the buffer to empty. Does NOT reseed the RNG."""
        self._obs.fill(0.0)
        self._actions.fill(0)
        self._rewards.fill(0.0)
        self._next_obs.fill(0.0)
        self._dones.fill(0.0)
        self._weights.fill(1.0)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def save(self, path: str | Path) -> None:
        """Persist buffer state (arrays + idx + size + RNG) to a compressed npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs=self._obs[: self._size],
            actions=self._actions[: self._size],
            rewards=self._rewards[: self._size],
            next_obs=self._next_obs[: self._size],
            dones=self._dones[: self._size],
            weights=self._weights[: self._size],
            idx=np.int64(self._idx),
            size=np.int64(self._size),
            capacity=np.int64(self.capacity),
            rng_state=np.array([self._rng.bit_generator.state], dtype=object),
        )

    def load(self, path: str | Path) -> None:
        """Restore buffer state. RNG state is restored so subsequent sample() byte-matches."""
        data = np.load(path, allow_pickle=True)
        size = int(data["size"])
        if int(data["capacity"]) != self.capacity:
            raise ValueError(
                f"capacity mismatch: file={int(data['capacity'])}, buffer={self.capacity}"
            )
        # Reset to known-empty then refill the live slice.
        self.clear()
        self._obs[:size] = data["obs"]
        self._actions[:size] = data["actions"]
        self._rewards[:size] = data["rewards"]
        self._next_obs[:size] = data["next_obs"]
        self._dones[:size] = data["dones"]
        self._weights[:size] = data["weights"]
        self._idx = int(data["idx"])
        self._size = size
        rng_state = data["rng_state"].item()
        self._rng.bit_generator.state = rng_state
