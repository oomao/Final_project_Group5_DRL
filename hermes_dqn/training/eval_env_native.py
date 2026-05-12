"""Apples-to-apples evaluation: greedy playback of a trained model on env-native reward.

Loads ``model_final.pt`` from a run dir, plays N greedy episodes against the
unshaped LunarLander-v3 env, and returns the metrics that gemma-reward-generator
and hermes-memory-layer use for cross-condition comparison. Eval seeds are
disjoint from training seeds (defaults to 10000+).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig


def evaluate_on_env_native(
    run_dir: str | Path,
    n: int = 100,
    base_seed: int = 10000,
    success_threshold: float = 200.0,
) -> dict[str, Any]:
    """Run ``n`` greedy episodes; return mean/success/crash metrics.

    The model and DQN config are loaded from ``run_dir``. The env is a fresh
    ``LunarLander-v3`` with no reward wrapper, so reported metrics are the
    env's native shaped return — directly comparable across training conditions.
    """
    run_dir = Path(run_dir)
    with (run_dir / "config.json").open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    dqn_cfg = DQNConfig(**cfg["dqn"])

    agent = DQNAgent(dqn_cfg, seed=cfg.get("seed", 42))
    agent.load(run_dir / "model_final.pt")

    env = gym.make("LunarLander-v3")
    env.action_space.seed(base_seed + n + 1)

    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(n):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_return = 0.0
        ep_length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_length += 1
        returns.append(ep_return)
        lengths.append(ep_length)
    env.close()

    arr = np.array(returns)
    return {
        "n": n,
        "base_seed": base_seed,
        "env_native_mean": float(arr.mean()),
        "env_native_median": float(np.median(arr)),
        "env_native_success": float((arr >= success_threshold).mean()),
        "env_native_crash_rate": float((arr < 0).mean()),
        "env_native_mean_length": float(np.mean(lengths)),
        "success_threshold": success_threshold,
    }
