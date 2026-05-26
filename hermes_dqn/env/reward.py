"""Reward-function plug-in contract.

Any callable matching :class:`RewardFunction` can be passed to
:func:`hermes_dqn.env.lunar_lander.make_env` to replace the env's native reward
without touching the agent or training loop. Later Hermes-generated rewards
will satisfy this Protocol by construction.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class RewardFunction(Protocol):
    """Structural type for swappable reward functions.

    Implementations receive the full transition plus the env's native reward,
    and must return a single float. Raising is allowed and will propagate to
    the training loop — see the reward-plugin spec for failure isolation.
    """

    def __call__(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        env_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> float: ...


def default_reward_fn(
    obs: np.ndarray,
    action: int,
    next_obs: np.ndarray,
    env_reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> float:
    """Passthrough — returns the env's native reward unchanged.

    Documented as a reference implementation. The env wrapper short-circuits
    when ``reward_fn is None``, so this function is rarely invoked at runtime;
    it exists so reward authors have a known-good template to copy.
    """
    return float(env_reward)
