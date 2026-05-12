"""Render side-by-side training curves as an animated GIF.

Loads episodes.jsonl from two runs and animates the per-episode return + a
100-episode rolling mean as the agents learn over time. Useful for slides
and demos to show convergence dynamics qualitatively.

Usage:
    python tools/animate_training.py <baseline-run-dir> <gemma-run-dir> [out.gif]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _load_returns(jsonl_path: Path) -> np.ndarray:
    with jsonl_path.open("r", encoding="utf-8") as fp:
        return np.array([json.loads(line)["return"] for line in fp])


def _rolling(arr: np.ndarray, window: int = 100) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    cumsum = np.cumsum(arr, dtype=float)
    for i in range(len(arr)):
        if i + 1 < window:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[i - window]) / window
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: animate_training.py <baseline-dir> <gemma-dir> [out.gif]", file=sys.stderr)
        return 1

    baseline_dir = Path(sys.argv[1])
    gemma_dir = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("runs/training_animation.gif")

    base_ret = _load_returns(baseline_dir / "episodes.jsonl")
    gemma_ret = _load_returns(gemma_dir / "episodes.jsonl")
    n = min(len(base_ret), len(gemma_ret))
    base_ret = base_ret[:n]
    gemma_ret = gemma_ret[:n]
    base_roll = _rolling(base_ret)
    gemma_roll = _rolling(gemma_ret)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("DQN Training Progress (seed=42, 1500 episodes)", fontsize=14, fontweight="bold")
    eps = np.arange(1, n + 1)

    # Static elements
    for ax, title in [(ax1, "Baseline (env-native reward)"), (ax2, "Gemma 4 31B-generated reward")]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_xlim(0, n)
        ax.axhline(200, ls="--", color="green", alpha=0.4, label="success threshold")
        ax.axhline(0, ls="-", color="black", alpha=0.2)
        ax.grid(alpha=0.3)
    ax1.set_ylabel("Return")
    ax1.set_ylim(min(base_ret.min(), gemma_ret.min()) - 50, max(base_ret.max(), gemma_ret.max()) + 50)

    # Animated lines
    (raw1,) = ax1.plot([], [], alpha=0.25, color="#1f77b4", lw=0.7, label="per-episode")
    (roll1,) = ax1.plot([], [], color="#1f77b4", lw=2.2, label="rolling-100 mean")
    (raw2,) = ax2.plot([], [], alpha=0.25, color="#d62728", lw=0.7, label="per-episode")
    (roll2,) = ax2.plot([], [], color="#d62728", lw=2.2, label="rolling-100 mean")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)

    progress_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10, fontweight="bold")

    # Sample frames at every 10 episodes for smoother / smaller output
    step = 10
    frames = list(range(step, n + 1, step))
    if frames[-1] != n:
        frames.append(n)

    def update(t: int):
        x = eps[:t]
        raw1.set_data(x, base_ret[:t])
        roll1.set_data(x, base_roll[:t])
        raw2.set_data(x, gemma_ret[:t])
        roll2.set_data(x, gemma_roll[:t])
        progress_text.set_text(
            f"Episode {t} / {n}   |   "
            f"baseline rolling: {base_roll[t-1]:+.1f}   "
            f"gemma rolling: {gemma_roll[t-1]:+.1f}"
        )
        return raw1, roll1, raw2, roll2, progress_text

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(frames)} frames -> {out_path} (this takes ~30-60s)...")
    ani.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"Done. File size: {out_path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
