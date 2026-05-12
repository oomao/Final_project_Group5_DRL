"""Record side-by-side gameplay of two trained models as a GIF.

Both agents play the same seeds, lock-stepped per env tick. The slower
agent dictates how long each episode contributes to the GIF; the faster
one shows its final frame until both are done.

Usage:
    python tools/compare_playback.py <left-dir> <right-dir> [out.gif]
        [--episodes N] [--seed BASE] [--fps F] [--every K]

Defaults: episodes=3, base seed=2000 (-> 2001..200N), fps=30, every=2 (sub-sample).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig


def _load_agent(run_dir: Path) -> DQNAgent:
    with (run_dir / "config.json").open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    dqn_cfg = DQNConfig(**cfg["dqn"])
    agent = DQNAgent(dqn_cfg, seed=42)
    agent.load(run_dir / "model_final.pt")
    return agent


def _annotate(
    frame: np.ndarray,
    left_label: str,
    right_label: str,
    ep: int,
    n_ep: int,
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
        font = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((10, 5), left_label, fill=(255, 255, 255), font=font)
    draw.text((half + 10, 5), right_label, fill=(255, 255, 255), font=font)
    state_l = "DONE" if left_done else "..."
    state_r = "DONE" if right_done else "..."
    draw.text(
        (10, 25),
        f"ep {ep}/{n_ep}  return {left_return:+.1f}  [{state_l}]",
        fill=(255, 255, 255),
        font=small,
    )
    draw.text(
        (half + 10, 25),
        f"ep {ep}/{n_ep}  return {right_return:+.1f}  [{state_r}]",
        fill=(255, 255, 255),
        font=small,
    )
    return np.array(img)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("left_dir", type=str, help="left-pane run directory (e.g., baseline)")
    p.add_argument("right_dir", type=str, help="right-pane run directory (e.g., Gemma)")
    p.add_argument("out_path", type=str, nargs="?", default="runs/comparison.gif")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=2000, help="base seed; uses seed+1..seed+N")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--every", type=int, default=2, help="keep every Kth frame to shrink GIF")
    p.add_argument("--left-label", type=str, default="BASELINE (env reward)")
    p.add_argument("--right-label", type=str, default="GEMMA reward")
    args = p.parse_args()

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_path = Path(args.out_path)

    left_agent = _load_agent(left_dir)
    right_agent = _load_agent(right_dir)

    left_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    right_env = gym.make("LunarLander-v3", render_mode="rgb_array")

    frames: list[np.ndarray] = []
    print(f"Recording {args.episodes} episodes (left: {left_dir.name}, right: {right_dir.name})...")
    for ep in range(1, args.episodes + 1):
        seed = args.seed + ep
        left_obs, _ = left_env.reset(seed=seed)
        right_obs, _ = right_env.reset(seed=seed)
        left_done = right_done = False
        left_return = right_return = 0.0
        last_left_frame = left_env.render()
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
            if tick % args.every == 0:
                combo = np.hstack([last_left_frame, last_right_frame])
                annotated = _annotate(
                    combo,
                    args.left_label,
                    args.right_label,
                    ep,
                    args.episodes,
                    left_return,
                    right_return,
                    left_done,
                    right_done,
                )
                frames.append(annotated)
            tick += 1
            # Safety cap
            if tick > 2000:
                break

        # Hold final frame for ~1 second so the result is readable
        hold_frames = max(args.fps, 30)
        final = frames[-1] if frames else None
        if final is not None:
            for _ in range(hold_frames):
                frames.append(final)
        print(
            f"  ep {ep}: left return {left_return:+.1f} ({'LANDED' if left_return >= 200 else 'crashed' if left_return < 0 else 'soft'}),"
            f" right return {right_return:+.1f} ({'LANDED' if right_return >= 200 else 'crashed' if right_return < 0 else 'soft'})"
        )

    left_env.close()
    right_env.close()

    print(f"Composing GIF: {len(frames)} frames @ {args.fps} fps -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / args.fps)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Done. File size: {out_path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
