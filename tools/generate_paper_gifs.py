"""Generate one B0-vs-Hermes side-by-side GIF per environment.

For each of the 4 classical-control envs (LunarLander, CartPole, MountainCar,
Acrobot), this:
  1. Loads the B0-env-native seed_42 trained model (left panel) and the
     B3-hermes-full seed_42 last-iter model (right panel).
  2. Plays both agents on the same evaluation seeds in lock-step.
  3. Composites a side-by-side annotated GIF.
  4. Saves to paper/gifs/<env_short>.gif.

Env-agnostic: env_id and DQN architecture (vanilla / dueling) are read from
each run's config.json — so this works for any of our 4 envs.

Usage:
    python tools/generate_paper_gifs.py
    python tools/generate_paper_gifs.py --envs LunarLander-v3,CartPole-v1
    python tools/generate_paper_gifs.py --episodes 3 --fps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `from hermes_dqn ...` imports when run as `python tools/generate_paper_gifs.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig


# ----------------------------------------------------------------------------
# Per-env metadata: which run-root holds the Part-1 results, plus a short name
# for the output GIF filename and the success-threshold language used in the
# overlay text.
# ----------------------------------------------------------------------------
ENV_META = {
    "LunarLander-v3": {
        "run_root": "runs/final",
        "short":    "lunarlander",
        "success_label": "LANDED",          # for env_native_mean >= threshold
        "success_threshold": 200.0,
        "fail_threshold":   -100.0,         # crash
        "fail_label":       "crashed",
    },
    "CartPole-v1": {
        "run_root": "runs/final_cp",
        "short":    "cartpole",
        "success_label": "SOLVED",
        "success_threshold": 475.0,
        "fail_threshold":   None,
        "fail_label":       None,
    },
    "MountainCar-v0": {
        "run_root": "runs/final_mc",
        "short":    "mountaincar",
        "success_label": "REACHED",
        "success_threshold": -110.0,        # standard "solved" mean
        "fail_threshold":   None,
        "fail_label":       None,
    },
    "Acrobot-v1": {
        "run_root": "runs/final_acr",
        "short":    "acrobot",
        "success_label": "SWUNG-UP",
        "success_threshold": -100.0,
        "fail_threshold":   None,
        "fail_label":       None,
    },
}


def _load_agent(run_dir: Path) -> DQNAgent:
    """Reconstruct the DQN agent from a run directory's config.json + model_final.pt."""
    with (run_dir / "config.json").open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    dqn_cfg = DQNConfig(**cfg["dqn"])
    agent = DQNAgent(dqn_cfg, seed=cfg.get("seed", 42))
    agent.load(run_dir / "model_final.pt")
    return agent


def _label_return(env_id: str, ret: float, done: bool) -> str:
    """Per-env return label that mirrors how each env defines success."""
    meta = ENV_META[env_id]
    if done:
        if ret >= meta["success_threshold"]:
            return meta["success_label"]
        if meta["fail_threshold"] is not None and ret < meta["fail_threshold"]:
            return meta["fail_label"]
        return "ended"
    return "..."


def _annotate(
    frame: np.ndarray,
    left_label: str,
    right_label: str,
    ep: int,
    n_ep: int,
    env_id: str,
    left_return: float,
    right_return: float,
    left_done: bool,
    right_done: bool,
) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, _ = img.size
    half = w // 2
    try:
        font_main = ImageFont.truetype("arial.ttf", 16)
        font_sub  = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font_main = ImageFont.load_default()
        font_sub  = ImageFont.load_default()
    draw.text((10, 5), left_label,  fill=(255, 255, 255), font=font_main)
    draw.text((half + 10, 5), right_label, fill=(255, 255, 255), font=font_main)
    l_state = _label_return(env_id, left_return,  left_done)
    r_state = _label_return(env_id, right_return, right_done)
    draw.text(
        (10, 25),
        f"ep {ep}/{n_ep}  return {left_return:+.1f}  [{l_state}]",
        fill=(255, 255, 255), font=font_sub,
    )
    draw.text(
        (half + 10, 25),
        f"ep {ep}/{n_ep}  return {right_return:+.1f}  [{r_state}]",
        fill=(255, 255, 255), font=font_sub,
    )
    return np.array(img)


def _resolve_run_dirs(env_id: str) -> tuple[Path, Path]:
    """Pick the B0 (left) and B3-hermes-full (right) run dirs for one env."""
    root = Path(ENV_META[env_id]["run_root"])
    b0 = root / "B0-env-native" / "seed_42" / "iter_01"
    # B3-hermes-full has 5 iters; use the last one (the closed loop's final output)
    b3_seed_dir = root / "B3-hermes-full" / "seed_42"
    iter_dirs = sorted(b3_seed_dir.glob("iter_*"))
    if not iter_dirs:
        raise FileNotFoundError(f"no iter_* under {b3_seed_dir}")
    b3 = iter_dirs[-1]
    return b0, b3


def _make_gif(
    env_id: str,
    out_path: Path,
    n_episodes: int = 3,
    base_seed: int = 5000,
    fps: int = 30,
    frame_every: int = 2,
    max_steps: int = 1000,
) -> None:
    left_dir, right_dir = _resolve_run_dirs(env_id)
    print(f"[{env_id}] left={left_dir.name} right={right_dir.name}")

    left_agent  = _load_agent(left_dir)
    right_agent = _load_agent(right_dir)

    left_env  = gym.make(env_id, render_mode="rgb_array")
    right_env = gym.make(env_id, render_mode="rgb_array")

    frames: list[np.ndarray] = []
    for ep in range(1, n_episodes + 1):
        seed = base_seed + ep
        left_obs,  _ = left_env.reset(seed=seed)
        right_obs, _ = right_env.reset(seed=seed)
        left_done = right_done = False
        left_return = right_return = 0.0
        last_left_frame  = left_env.render()
        last_right_frame = right_env.render()

        tick = 0
        while not (left_done and right_done):
            if not left_done:
                a = left_agent.act(left_obs, epsilon=0.0)
                left_obs, r, term, trunc, _ = left_env.step(a)
                left_return += r
                left_done = term or trunc
                last_left_frame = left_env.render()
            if not right_done:
                a = right_agent.act(right_obs, epsilon=0.0)
                right_obs, r, term, trunc, _ = right_env.step(a)
                right_return += r
                right_done = term or trunc
                last_right_frame = right_env.render()
            if tick % frame_every == 0:
                combo = np.hstack([last_left_frame, last_right_frame])
                annotated = _annotate(
                    combo,
                    "B0-env-native",
                    "B3-hermes-full (Ours)",
                    ep, n_episodes,
                    env_id,
                    left_return, right_return,
                    left_done, right_done,
                )
                frames.append(annotated)
            tick += 1
            if tick > max_steps:
                break

        # Hold the final frame ~1s so the outcome is readable
        hold = max(fps, 30)
        if frames:
            for _ in range(hold):
                frames.append(frames[-1])
        print(
            f"  ep {ep}: B0 return {left_return:+.1f} "
            f"[{_label_return(env_id, left_return, True)}], "
            f"Hermes return {right_return:+.1f} "
            f"[{_label_return(env_id, right_return, True)}]"
        )

    left_env.close()
    right_env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  -> {out_path}  ({out_path.stat().st_size / 1024:.1f} KB, {len(frames)} frames)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--envs",
        type=str,
        default=",".join(ENV_META.keys()),
        help="comma-separated env_ids to render (default: all 4)",
    )
    p.add_argument("--out-dir", type=str, default="paper/gifs")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=5000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-every", type=int, default=2)
    args = p.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for env_id in envs:
        if env_id not in ENV_META:
            print(f"[skip] unknown env: {env_id}")
            continue
        short = ENV_META[env_id]["short"]
        out_path = out_dir / f"{short}.gif"
        _make_gif(
            env_id,
            out_path,
            n_episodes=args.episodes,
            base_seed=args.base_seed,
            fps=args.fps,
            frame_every=args.frame_every,
        )
    print(f"\nAll GIFs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
