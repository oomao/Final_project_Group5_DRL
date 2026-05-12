"""Watch a trained DQN agent play LunarLander-v3.

Usage:
    python -m hermes_dqn.training.play --run-dir runs/baseline_seed42
    python -m hermes_dqn.training.play --run-dir runs/baseline_seed42 --episodes 5 --epsilon 0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig
from hermes_dqn.env.lunar_lander import make_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a trained DQN agent playing LunarLander-v3.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a runs/<name> dir containing model_final.pt and config.json",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="How many episodes to render (default 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        help="Base env seed (offset per episode). Different from training seeds.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration during playback; 0.0 = pure greedy (default).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    model_path = run_dir / "model_final.pt"
    if not config_path.exists() or not model_path.exists():
        raise SystemExit(f"Missing config.json or model_final.pt in {run_dir}")

    with config_path.open("r", encoding="utf-8") as fp:
        config_data = json.load(fp)
    dqn_cfg = DQNConfig(**config_data["dqn"])

    env = make_env(seed=args.seed, render_mode="human")
    agent = DQNAgent(dqn_cfg, seed=args.seed)
    agent.load(model_path)

    returns: list[float] = []
    print(f"Playing {args.episodes} episodes from {run_dir} (epsilon={args.epsilon})")
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, epsilon=args.epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            length += 1
        returns.append(ep_return)
        outcome = "LANDED" if ep_return >= 200 else ("crashed" if ep_return < 0 else "soft-land")
        print(f"  Episode {ep}: return = {ep_return:6.1f}  length = {length:4d}  [{outcome}]")

    env.close()

    mean = sum(returns) / len(returns)
    success_rate = sum(1 for r in returns if r >= 200) / len(returns)
    print(f"\n[OK] Played {args.episodes} episodes.")
    print(f"  Mean return    : {mean:.1f}")
    print(f"  Success rate   : {success_rate:.0%}  (return >= 200)")


if __name__ == "__main__":
    main()
