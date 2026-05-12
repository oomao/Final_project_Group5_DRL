"""Fair comparison: evaluate a trained model on env-native reward.

Loads model_final.pt from a run dir, plays N greedy episodes against the
unshaped LunarLander-v3 env, and reports mean reward + success rate.
Use distinct eval seeds (10000+) from training seeds.

Usage:
    python tools/_eval_env_native.py <run-dir> [n_episodes]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _eval_env_native.py <run-dir> [n_episodes]", file=sys.stderr)
        return 1
    run_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) >= 3 else 100

    with (run_dir / "config.json").open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    dqn_cfg = DQNConfig(**cfg["dqn"])

    agent = DQNAgent(dqn_cfg, seed=42)
    agent.load(run_dir / "model_final.pt")

    env = gym.make("LunarLander-v3")
    env.action_space.seed(9999)

    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(n):
        obs, _ = env.reset(seed=10000 + ep)
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
    success_rate = float((arr >= 200.0).mean())
    print(f"Run dir              : {run_dir}")
    print(f"reward_source        : {cfg.get('reward_source', 'unknown')}")
    print(f"Episodes evaluated   : {n}  (greedy, eval seeds 10000..{10000 + n - 1})")
    print(f"Mean env-native rwd  : {arr.mean():.2f}  (median {np.median(arr):.2f})")
    print(f"Success rate (>=200) : {success_rate:.0%}")
    print(f"Crash rate (<0)      : {(arr < 0).mean():.0%}")
    print(f"Mean ep length       : {np.mean(lengths):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
