"""Variance-fingerprint GIF: same condition (B3-hermes-full), different seeds.
Top row  MountainCar (std 3.08, seeds near-identical = stable).
Bottom   LunarLander (std 91.4, seeds diverge = chaotic).
English on-canvas labels (slide caption is Chinese).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.generate_presentation_gifs import (
    load_agent, save_gif, _seed_dir, _last_iter, play_panels, ENV_ID,
)

SEEDS = [42, 44, 46]
EVAL, MAX, ROW_W, BAR = 7000, 500, 1140, 30


def font(sz):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def build_row(short, label):
    env_id = ENV_ID[short]
    agents = [load_agent(_last_iter(_seed_dir(short, "vanilla", "B3-hermes-full", s))) for s in SEEDS]
    panels, _ = play_panels(agents, env_id, EVAL, frame_every=2, max_steps=MAX)
    frames = []
    for frame, _r, _d in panels:
        h, w, _ = frame.shape
        rh = int(round(h * ROW_W / w))
        small = np.asarray(Image.fromarray(frame).resize((ROW_W, rh), Image.BILINEAR))
        canvas = np.full((rh + BAR, ROW_W, 3), 245, np.uint8)
        canvas[BAR:] = small
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        d.text((10, 6), label, font=font(18), fill=(10, 10, 10))
        seg = ROW_W // len(SEEDS)
        for i, s in enumerate(SEEDS):
            if i:
                d.line([(i * seg, BAR), (i * seg, rh + BAR)], fill=(205, 205, 205), width=2)
            d.text((i * seg + 8, BAR + 4), f"seed {s}", font=font(13), fill=(70, 70, 70))
        frames.append(np.asarray(im))
    return frames


r1 = build_row("mc", "MountainCar  -  sparse  -  std 3.08  (seeds nearly identical = stable)")
r2 = build_row("ll", "LunarLander  -  dense  -  std 91.4  (seeds diverge = chaotic)")
n = max(len(r1), len(r2))
r1 += [r1[-1]] * (n - len(r1))
r2 += [r2[-1]] * (n - len(r2))
gap = np.full((10, ROW_W, 3), 245, np.uint8)
frames = [np.vstack([r1[k], gap, r2[k]]) for k in range(n)]
save_gif(frames, Path("paper/gifs/presentation/variance_fingerprint.gif"), fps=24, hold_last=20)
print("DONE variance_fingerprint.gif", len(frames), "frames")
