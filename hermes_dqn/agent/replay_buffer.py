"""Uniform replay buffer backed by numpy circular arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer over numpy arrays.

    Pre-allocates contiguous storage so ``push`` is O(1) and ``sample`` is a
    single fancy-indexed read. Internally uses ``float32`` for observations
    and rewards to keep the torch hand-off cheap.
    """

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
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
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            dones=self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size
