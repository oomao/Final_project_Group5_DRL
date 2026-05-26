"""LunarLander-v3 factory with injectable reward function."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from hermes_dqn.env.reward import RewardFunction


class RewardInjectionWrapper(gym.Wrapper):
    """Replaces the env's native reward with ``reward_fn`` if one is provided.

    On every ``step()``, the wrapper passes the pre-step observation, the
    action just taken, the post-step observation, the env's native reward,
    the done flags, and the env's info dict to ``reward_fn``. The return
    value becomes the reward seen by the agent. If ``reward_fn is None``,
    the env's native reward is returned unchanged.
    """

    def __init__(self, env: gym.Env, reward_fn: RewardFunction | None = None):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._last_obs: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)
        if self.reward_fn is None:
            reward = float(env_reward)
        else:
            assert self._last_obs is not None, "step() called before reset()"
            reward = float(
                self.reward_fn(
                    self._last_obs,
                    int(action),
                    next_obs,
                    float(env_reward),
                    bool(terminated),
                    bool(truncated),
                    info,
                )
            )
        self._last_obs = next_obs
        return next_obs, reward, terminated, truncated, info


def make_env(
    seed: int,
    reward_fn: RewardFunction | None = None,
    env_id: str = "LunarLander-v3",
    render_mode: str | None = None,
) -> gym.Env:
    """Construct a seeded LunarLander env, optionally with a custom reward.

    Seeding is applied to both ``env.reset(seed=...)`` (state RNG) and
    ``env.action_space.seed(seed)`` (sampling RNG) so the wrapper is fully
    reproducible. Reseeding the wrapper later via ``env.reset(seed=...)`` will
    reseed the underlying env as Gymnasium normally allows.

    Pass ``render_mode="human"`` for an interactive pygame window (for playback
    or live demos); leave ``None`` for headless training.
    """
    base_env = gym.make(env_id, render_mode=render_mode)
    wrapped = RewardInjectionWrapper(base_env, reward_fn=reward_fn)
    wrapped.reset(seed=seed)
    wrapped.action_space.seed(seed)
    return wrapped
