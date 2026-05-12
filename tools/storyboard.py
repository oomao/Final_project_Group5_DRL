"""Generate the headline visualization: 4-way comparison across the project's
training arc (env baseline -> Gemma -> Gemma+memory -> full closed-loop).

Outputs three artifacts to ``runs/`` / ``reports/storyboard/``:

1. ``runs/storyboard.gif``: 2x2 grid of trained agents playing the same seeds
   in lock-step (so each panel is at the same timestep / starting state).
2. ``reports/storyboard/training_curves.png``: per-condition reward curve
   with 100-ep rolling mean overlaid (one line per condition, all 1500 eps).
3. stdout summary table comparing env_native_mean / success / crash / wall_time
   across the four conditions.

Designed for talking through what the project does at the demo / oral exam.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig

# Order matters: GIF panels go top-left, top-right, bottom-left, bottom-right.
# Each tuple is (display_label, run_dir, color_for_curves).
RUNS = [
    ("Baseline\n(env reward)", "runs/baseline_seed42", "#1f77b4"),
    ("Gemma\n(no memory)", "runs/gemma_seed42", "#d62728"),
    ("Gemma\n+ memory", "runs/gemma_mem_seed42", "#2ca02c"),
    ("Full Hermes-DQN\n(closed loop iter 3)", "runs/pilot/B3-pilot/seed_42/iter_03", "#9467bd"),
]

# Visualization knobs
N_EPISODES = 3
BASE_SEED = 3000
FPS = 30
FRAME_EVERY = 2  # keep every Kth frame to shrink GIF


def _load_agent(run_dir: Path) -> DQNAgent:
    with (run_dir / "config.json").open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    dqn_cfg = DQNConfig(**cfg["dqn"])
    agent = DQNAgent(dqn_cfg, seed=42)
    agent.load(run_dir / "model_final.pt")
    return agent


def _load_returns(run_dir: Path) -> np.ndarray:
    jsonl = run_dir / "episodes.jsonl"
    if not jsonl.exists():
        return np.array([])
    with jsonl.open("r", encoding="utf-8") as fp:
        return np.array([json.loads(line)["return"] for line in fp])


def _rolling100(arr: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    csum = np.cumsum(arr, dtype=float)
    for i in range(len(arr)):
        if i + 1 < 100:
            out[i] = csum[i] / (i + 1)
        else:
            out[i] = (csum[i] - csum[i - 100]) / 100
    return out


def _load_env_native_stats(run_dir: Path) -> dict[str, float | None]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    if cfg.get("env_native_mean") is not None:
        return {
            "env_native_mean": cfg["env_native_mean"],
            "env_native_success": cfg["env_native_success"],
            "env_native_crash_rate": cfg.get("env_native_crash_rate"),
            "env_native_mean_length": cfg.get("env_native_mean_length"),
        }
    # Fallback: legacy runs (pre hermes-memory-layer inline eval) — compute now
    from hermes_dqn.training.eval_env_native import evaluate_on_env_native

    print(f"  [fallback] {run_dir.name}: env-native eval not cached, computing 100-seed eval...")
    metrics = evaluate_on_env_native(run_dir, n=100, base_seed=10000)
    # Cache back to config.json for future reuse
    cfg["env_native_mean"] = metrics["env_native_mean"]
    cfg["env_native_success"] = metrics["env_native_success"]
    cfg["env_native_crash_rate"] = metrics["env_native_crash_rate"]
    cfg["env_native_mean_length"] = metrics["env_native_mean_length"]
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2)
    return {
        "env_native_mean": metrics["env_native_mean"],
        "env_native_success": metrics["env_native_success"],
        "env_native_crash_rate": metrics["env_native_crash_rate"],
        "env_native_mean_length": metrics["env_native_mean_length"],
    }


def _annotate_panel(
    frame: np.ndarray,
    label: str,
    ep: int,
    n_ep: int,
    cumulative_return: float,
    done: bool,
) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    # Label box (top-left)
    draw.rectangle([(0, 0), (img.size[0], 38)], fill=(0, 0, 0, 180))
    draw.text((6, 2), label, fill=(255, 255, 255), font=title_font)
    state = "DONE" if done else "..."
    if done:
        outcome = "LANDED" if cumulative_return >= 200 else ("crashed" if cumulative_return < 0 else "soft")
        state = f"DONE [{outcome}]"
    draw.text(
        (6, 20),
        f"ep {ep}/{n_ep}  return {cumulative_return:+.1f}  {state}",
        fill=(255, 255, 255),
        font=small_font,
    )
    return np.array(img)


def render_storyboard_gif(out_path: Path) -> None:
    agents = [_load_agent(Path(r[1])) for r in RUNS]
    envs = [gym.make("LunarLander-v3", render_mode="rgb_array") for _ in RUNS]
    for env in envs:
        env.action_space.seed(BASE_SEED + 9999)

    print(f"Rendering {N_EPISODES} lock-stepped episodes...")
    composite_frames: list[np.ndarray] = []

    for ep in range(1, N_EPISODES + 1):
        seed = BASE_SEED + ep
        obs_list = [env.reset(seed=seed)[0] for env in envs]
        returns = [0.0] * len(RUNS)
        dones = [False] * len(RUNS)
        last_frames = [env.render() for env in envs]

        tick = 0
        while not all(dones):
            for i in range(len(RUNS)):
                if not dones[i]:
                    action = agents[i].act(obs_list[i], epsilon=0.0)
                    obs_list[i], r, term, trunc, _ = envs[i].step(action)
                    returns[i] += r
                    dones[i] = term or trunc
                    last_frames[i] = envs[i].render()

            if tick % FRAME_EVERY == 0:
                panels = [
                    _annotate_panel(
                        last_frames[i],
                        RUNS[i][0],
                        ep,
                        N_EPISODES,
                        returns[i],
                        dones[i],
                    )
                    for i in range(len(RUNS))
                ]
                top = np.hstack([panels[0], panels[1]])
                bot = np.hstack([panels[2], panels[3]])
                grid = np.vstack([top, bot])
                composite_frames.append(grid)

            tick += 1
            if tick > 2500:
                break

        # Hold final 1 second so result is readable
        hold = max(FPS, 30)
        if composite_frames:
            for _ in range(hold):
                composite_frames.append(composite_frames[-1])

        print(f"  ep {ep}: " + " | ".join(f"{RUNS[i][0].splitlines()[0]}={returns[i]:+.1f}" for i in range(4)))

    for env in envs:
        env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Composing GIF: {len(composite_frames)} frames @ {FPS} fps -> {out_path}")
    duration = int(1000 / FPS)
    pil_frames = [Image.fromarray(f) for f in composite_frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    print(f"  done. {out_path.stat().st_size / 1024 / 1024:.1f} MB")


def render_training_curves(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
    for label, run_dir, color in RUNS:
        ret = _load_returns(Path(run_dir))
        if ret.size == 0:
            continue
        rolling = _rolling100(ret)
        x = np.arange(1, len(ret) + 1)
        # Faint raw line + thick rolling mean
        ax.plot(x, ret, color=color, alpha=0.10, linewidth=0.6)
        clean_label = label.replace("\n", " ")
        ax.plot(x, rolling, color=color, linewidth=2.2, label=clean_label)
    ax.axhline(200, ls="--", color="green", alpha=0.4, label="success threshold (env reward = 200)")
    ax.axhline(0, ls="-", color="black", alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (per episode + 100-ep rolling mean)")
    ax.set_title(
        "Hermes-DQN training arc on LunarLander-v3 (seed 42)\n"
        "shaped return varies by reward source — see env-native eval table for apples-to-apples"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  done. {out_path.stat().st_size / 1024:.1f} KB")


def print_summary_table() -> None:
    print("\n=== Apples-to-apples env-native eval (greedy on 100 unseen seeds 10000-10099) ===")
    print(
        f"{'Condition':<38}  {'env_mean':>9}  {'success':>8}  {'crash':>7}  {'mean_len':>8}"
    )
    print("-" * 80)
    for label, run_dir, _ in RUNS:
        stats = _load_env_native_stats(Path(run_dir))
        clean = label.replace("\n", " ")
        env_mean = stats["env_native_mean"]
        if env_mean is None:
            print(f"{clean:<38}  (no env-native eval recorded)")
            continue
        print(
            f"{clean:<38}  {env_mean:>+9.2f}  "
            f"{stats['env_native_success']:>7.2%}  "
            f"{stats['env_native_crash_rate']:>6.2%}  "
            f"{stats['env_native_mean_length']:>8.0f}"
        )


def main() -> int:
    out_root = Path("reports/storyboard")
    out_root.mkdir(parents=True, exist_ok=True)

    print_summary_table()
    print()
    render_training_curves(out_root / "training_curves.png")
    render_storyboard_gif(Path("runs") / "storyboard.gif")

    print("\n=== Outputs ===")
    print(f"  GIF:    runs/storyboard.gif")
    print(f"  Curves: reports/storyboard/training_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
