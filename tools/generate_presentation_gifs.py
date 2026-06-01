"""Presentation-ready GIFs that are *consistent with the reported statistics*.

The paper tables report 5-seed means (each seed's number is itself a 100-episode
eval mean). A naive playback GIF shows one arbitrary episode of one arbitrary
seed, which can disagree badly with those headline numbers (e.g. Dueling
CartPole seed_42 scores 129 vs the 5-seed mean of 316). To stay honest on a
slide, this script picks *representative* seeds and *representative* episodes:

  * representative seed   -> the seed whose env_native_mean is closest to the
                            5-seed mean (median-like, never the lucky best).
  * representative episode -> among candidate eval seeds, the episode whose
                            return is closest to that model's own mean.

Three deliverables (all written to paper/gifs/presentation/):

  1. env  -> per-env B0-env-native vs B3-hermes-full side-by-side, with a footer
            carrying the 5-seed aggregate (mean +/- std, delta%, Mann-Whitney p).
  2. grid -> one 2x2 GIF tiling all four envs (each cell = that env's B0|B3),
            for one-click presentation.
  3. compare -> one 3-panel Vanilla|Double|Dueling GIF (B3-hermes agents) for a
            single representative env (default CartPole), each panel on its
            representative seed so the visual matches Table 6's
            "architecture-agnostic" finding.

Usage:
    python tools/generate_presentation_gifs.py                 # all three
    python tools/generate_presentation_gifs.py --task env
    python tools/generate_presentation_gifs.py --task compare --compare-env cp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import mannwhitneyu

from hermes_dqn.agent.dqn_agent import DQNAgent, DQNConfig

# ----------------------------------------------------------------------------
# Static metadata
# ----------------------------------------------------------------------------
ENV_ID = {
    "ll": "LunarLander-v3",
    "cp": "CartPole-v1",
    "mc": "MountainCar-v0",
    "acr": "Acrobot-v1",
}
VANILLA_DIR = {"cp": "final_cp", "mc": "final_mc", "acr": "final_acr", "ll": "final"}
SEEDS = [42, 43, 44, 45, 46]
VARIANT_LABEL = {"vanilla": "Vanilla DQN", "double": "Double DQN", "dueling": "Dueling DQN"}
GRID_ORDER = ["ll", "cp", "mc", "acr"]
RUNS = Path("runs")

# per-env success / failure language for the on-screen state tag
STATE = {
    "CartPole-v1":    dict(succ=475.0, succ_lbl="SOLVED",   fail=None,   fail_lbl=None),
    "LunarLander-v3": dict(succ=200.0, succ_lbl="LANDED",   fail=-100.0, fail_lbl="CRASHED"),
    "MountainCar-v0": dict(succ=-199.5, succ_lbl="REACHED", fail=None,   fail_lbl=None),
    "Acrobot-v1":     dict(succ=-100.0, succ_lbl="SWUNG-UP", fail=None,  fail_lbl=None),
}


def _font(size: int, bold: bool = False):
    for name in ((["arialbd.ttf", "arial.ttf"]) if bold else ["arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_H = _font(17, bold=True)
FONT_S = _font(13)
FONT_F = _font(13)
FONT_CELL = _font(13, bold=True)


# ----------------------------------------------------------------------------
# Run-dir resolution + stats
# ----------------------------------------------------------------------------
def _variant_root(short: str, variant: str) -> Path:
    if variant == "vanilla":
        return RUNS / VANILLA_DIR[short]
    return RUNS / f"part2_{variant}_{short}"


def _seed_dir(short: str, variant: str, cond: str, seed: int) -> Path:
    return _variant_root(short, variant) / cond / f"seed_{seed}"


def _last_iter(seed_dir: Path) -> Path:
    iters = sorted(seed_dir.glob("iter_*"))
    if not iters:
        raise FileNotFoundError(f"no iter_* under {seed_dir}")
    return iters[-1]  # B0 has only iter_01; B3 -> final closed-loop iter


def _seed_mean(seed_dir: Path) -> float:
    cfg = json.loads((_last_iter(seed_dir) / "config.json").read_text(encoding="utf-8"))
    return float(cfg["env_native_mean"])


def aggregate(short: str, variant: str, cond: str):
    """Return (per_seed: dict seed->mean, mean, std)."""
    per = {s: _seed_mean(_seed_dir(short, variant, cond, s)) for s in SEEDS}
    vals = np.array(list(per.values()), dtype=float)
    return per, float(vals.mean()), float(vals.std(ddof=1))


def pair_stats(short: str, variant: str) -> dict:
    b0p, b0m, b0s = aggregate(short, variant, "B0-env-native")
    b3p, b3m, b3s = aggregate(short, variant, "B3-hermes-full")
    delta = (b3m - b0m) / abs(b0m) * 100.0
    _, p = mannwhitneyu(list(b3p.values()), list(b0p.values()), alternative="two-sided")
    return dict(b0p=b0p, b0m=b0m, b0s=b0s, b3p=b3p, b3m=b3m, b3s=b3s, delta=delta, p=float(p))


def representative(short: str, variant: str, cond: str):
    """Seed whose mean is closest to the 5-seed mean, plus its run dir + mean."""
    per, mean, _ = aggregate(short, variant, cond)
    seed = min(per, key=lambda s: abs(per[s] - mean))
    return seed, _last_iter(_seed_dir(short, variant, cond, seed)), per[seed]


# ----------------------------------------------------------------------------
# Agent loading + rollouts
# ----------------------------------------------------------------------------
def load_agent(run_dir: Path) -> DQNAgent:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    agent = DQNAgent(DQNConfig(**cfg["dqn"]), seed=int(cfg.get("seed", 42)))
    agent.load(run_dir / "model_final.pt")
    return agent


def score(agent: DQNAgent, env_id: str, seed: int, max_steps: int) -> float:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    ret, done, steps = 0.0, False, 0
    while not done and steps < max_steps:
        obs, r, term, trunc, _ = env.step(agent.act(obs, epsilon=0.0))
        ret += r
        done = term or trunc
        steps += 1
    env.close()
    return ret


def pick_episode_seeds(agents, env_id, targets, candidates, max_steps, k):
    """Pick k episode seeds whose per-agent returns sit closest to `targets`."""
    scored = []
    for s in candidates:
        rets = [score(a, env_id, s, max_steps) for a in agents]
        dev = sum(abs(rets[i] - targets[i]) / (abs(targets[i]) + 1.0) for i in range(len(agents)))
        scored.append((dev, s))
    scored.sort()
    return [s for _, s in scored[:k]]


def play_panels(agents, env_id, seed, frame_every, max_steps):
    """Roll out N agents on one shared env seed; yield hstacked render frames."""
    envs = [gym.make(env_id, render_mode="rgb_array") for _ in agents]
    obs = [e.reset(seed=seed)[0] for e in envs]
    done = [False] * len(agents)
    ret = [0.0] * len(agents)
    last = [e.render() for e in envs]
    out, tick = [], 0
    while not all(done) and tick <= max_steps:
        for i, (ag, e) in enumerate(zip(agents, envs)):
            if not done[i]:
                obs[i], r, term, trunc, _ = e.step(ag.act(obs[i], epsilon=0.0))
                ret[i] += r
                done[i] = term or trunc
                last[i] = e.render()
        if tick % frame_every == 0:
            out.append((np.hstack(last), list(ret), list(done)))
        tick += 1
    for e in envs:
        e.close()
    return out, ret


def state_tag(env_id: str, ret: float, done: bool) -> str:
    if not done:
        return "..."
    m = STATE[env_id]
    if ret >= m["succ"]:
        return m["succ_lbl"]
    if m["fail"] is not None and ret <= m["fail"]:
        return m["fail_lbl"]
    return "timed out" if env_id == "MountainCar-v0" else "ended"


# ----------------------------------------------------------------------------
# Compositing
# ----------------------------------------------------------------------------
def compose(frame, headers, rets, states, ep, n_ep, footer, top_h=48, bot_h=30):
    h, w, _ = frame.shape
    canvas = np.full((h + top_h + bot_h, w, 3), 18, np.uint8)
    canvas[top_h:top_h + h] = frame
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    n = len(headers)
    seg = w // n
    for i in range(n):
        x = i * seg + 10
        d.text((x, 6), headers[i], font=FONT_H, fill=(255, 255, 255))
        d.text((x, 27), f"ep {ep}/{n_ep}  return {rets[i]:+.0f}  [{states[i]}]",
               font=FONT_S, fill=(210, 210, 210))
        if i:
            d.line([(i * seg, top_h), (i * seg, top_h + h)], fill=(70, 70, 70), width=2)
    d.text((10, h + top_h + 7), footer, font=FONT_F, fill=(235, 235, 235))
    return np.array(img)


def save_gif(frames, out_path: Path, fps: int, hold_last: int = 30):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(f) for f in frames]
    pil += [pil[-1]] * hold_last
    pil[0].save(out_path, save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  -> {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(pil)} frames)")


# ----------------------------------------------------------------------------
# Deliverable 1: per-env B0 vs B3 with aggregate footer
# ----------------------------------------------------------------------------
def build_env_gif(short, out_dir, variant="vanilla", n_episodes=2,
                  candidates=14, fps=30, frame_every=2, max_steps=600):
    env_id = ENV_ID[short]
    st = pair_stats(short, variant)
    _, b0_dir, b0_mean = representative(short, variant, "B0-env-native")
    _, b3_dir, b3_mean = representative(short, variant, "B3-hermes-full")
    print(f"[{env_id}/{variant}] B0 rep mean {b0_mean:.1f}, B3 rep mean {b3_mean:.1f}")

    agents = [load_agent(b0_dir), load_agent(b3_dir)]
    cand = list(range(7000, 7000 + candidates))
    seeds = pick_episode_seeds(agents, env_id, [b0_mean, b3_mean], cand, max_steps, n_episodes)

    footer = (f"{env_id}  |  5-seed env-native return:  "
              f"B0 {st['b0m']:.0f}+/-{st['b0s']:.0f}  vs  "
              f"B3-Hermes {st['b3m']:.0f}+/-{st['b3s']:.0f}  |  "
              f"delta {st['delta']:+.0f}%  |  Mann-Whitney p={st['p']:.3f}")

    frames = []
    for ep, s in enumerate(seeds, 1):
        panels, rets = play_panels(agents, env_id, s, frame_every, max_steps)
        for img, rsnap, dsnap in panels:
            frames.append(compose(
                img,
                ["B0-env-native", "B3-hermes-full (Ours)"],
                rsnap,
                [state_tag(env_id, rsnap[0], dsnap[0]), state_tag(env_id, rsnap[1], dsnap[1])],
                ep, n_episodes, footer))
        # freeze the outcome for ~0.7 s
        for img, rsnap, dsnap in [panels[-1]] * int(fps * 0.7):
            frames.append(compose(
                img,
                ["B0-env-native", "B3-hermes-full (Ours)"],
                [rets[0], rets[1]],
                [state_tag(env_id, rets[0], True), state_tag(env_id, rets[1], True)],
                ep, n_episodes, footer))
        print(f"  ep {ep} seed {s}: B0 {rets[0]:+.0f}, B3 {rets[1]:+.0f}")

    save_gif(frames, out_dir / f"{short}.gif", fps)
    return st


# ----------------------------------------------------------------------------
# Deliverable 2: 2x2 grid of the four envs
# ----------------------------------------------------------------------------
def build_grid_gif(out_dir, variant="vanilla", fps=24, frame_every=3,
                   max_steps=600, cell_w=480, candidates=14):
    cells = []  # list of (resized_frame_list, label)
    n_max = 0
    for short in GRID_ORDER:
        env_id = ENV_ID[short]
        st = pair_stats(short, variant)
        _, b0_dir, b0_mean = representative(short, variant, "B0-env-native")
        _, b3_dir, b3_mean = representative(short, variant, "B3-hermes-full")
        agents = [load_agent(b0_dir), load_agent(b3_dir)]
        cand = list(range(7000, 7000 + candidates))
        seed = pick_episode_seeds(agents, env_id, [b0_mean, b3_mean], cand, max_steps, 1)[0]
        panels, rets = play_panels(agents, env_id, seed, frame_every, max_steps)
        label = f"{env_id.split('-')[0]}   B0 | B3   delta{st['delta']:+.0f}%  p={st['p']:.3f}"

        # resize each side-by-side frame to a fixed cell width, add a label strip
        strip_h = 24
        resized = []
        for img, _r, _d in panels:
            h, w, _ = img.shape
            ch = int(round(h * cell_w / w))
            small = np.asarray(Image.fromarray(img).resize((cell_w, ch), Image.BILINEAR))
            cell = np.full((ch + strip_h, cell_w, 3), 18, np.uint8)
            cell[strip_h:] = small
            pim = Image.fromarray(cell)
            ImageDraw.Draw(pim).text((6, 5), label, font=FONT_CELL, fill=(245, 245, 245))
            resized.append(np.asarray(pim))
        cells.append(resized)
        n_max = max(n_max, len(resized))
        print(f"  grid cell {env_id}: seed {seed}  B0 {rets[0]:+.0f}  B3 {rets[1]:+.0f}  ({len(resized)} frames)")

    # pad every cell to the same length, then tile 2x2
    cell_shape = cells[0][0].shape
    for c in cells:
        if c[0].shape != cell_shape:  # heights differ across envs -> letterbox
            for i, f in enumerate(c):
                if f.shape != cell_shape:
                    pad = np.full(cell_shape, 18, np.uint8)
                    pad[:f.shape[0], :f.shape[1]] = f[:cell_shape[0], :cell_shape[1]]
                    c[i] = pad
        c += [c[-1]] * (n_max - len(c))

    frames = []
    for k in range(n_max):
        top = np.hstack([cells[0][k], cells[1][k]])
        bot = np.hstack([cells[2][k], cells[3][k]])
        frames.append(np.vstack([top, bot]))
    save_gif(frames, out_dir / "grid_2x2.gif", fps)


# ----------------------------------------------------------------------------
# Deliverable 3: 3-panel Vanilla | Double | Dueling for one env
# ----------------------------------------------------------------------------
def build_compare_gif(short, out_dir, n_episodes=2, candidates=14,
                      fps=30, frame_every=2, max_steps=600):
    env_id = ENV_ID[short]
    variants = ["vanilla", "double", "dueling"]
    agents, means, vmeans = [], [], []
    for v in variants:
        _, b3_dir, b3_mean = representative(short, v, "B3-hermes-full")
        agents.append(load_agent(b3_dir))
        means.append(b3_mean)
        vmeans.append(pair_stats(short, v)["b3m"])
    print(f"[compare {env_id}] rep means {[round(m,1) for m in means]}")

    cand = list(range(7000, 7000 + candidates))
    seeds = pick_episode_seeds(agents, env_id, means, cand, max_steps, n_episodes)

    footer = (f"{env_id}  |  Hermes-DQN env-native return (5-seed mean):  "
              f"Vanilla {vmeans[0]:.0f}  .  Double {vmeans[1]:.0f}  .  Dueling {vmeans[2]:.0f}"
              f"  |  pairwise p>0.30 (architecture-agnostic)")
    heads = [VARIANT_LABEL[v] + " (Hermes)" for v in variants]

    frames = []
    for ep, s in enumerate(seeds, 1):
        panels, rets = play_panels(agents, env_id, s, frame_every, max_steps)
        for img, rsnap, dsnap in panels:
            frames.append(compose(img, heads, rsnap,
                                   [state_tag(env_id, rsnap[i], dsnap[i]) for i in range(3)],
                                   ep, n_episodes, footer))
        for _ in range(int(fps * 0.7)):
            img = panels[-1][0]
            frames.append(compose(img, heads, rets,
                                   [state_tag(env_id, rets[i], True) for i in range(3)],
                                   ep, n_episodes, footer))
        print(f"  ep {ep} seed {s}: " + ", ".join(f"{VARIANT_LABEL[v]} {rets[i]:+.0f}"
                                                   for i, v in enumerate(variants)))
    save_gif(frames, out_dir / f"dqn_compare_{short}.gif", fps)


# ----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["all", "env", "grid", "compare"], default="all")
    p.add_argument("--envs", default="ll,cp,mc,acr", help="shorts for the per-env task")
    p.add_argument("--compare-env", default="cp", help="short for the 3-DQN comparison")
    p.add_argument("--variant", default="vanilla")
    p.add_argument("--out-dir", default="paper/gifs/presentation")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "env"):
        print("== per-env B0 vs B3 GIFs ==")
        for short in [e.strip() for e in args.envs.split(",") if e.strip()]:
            build_env_gif(short, out_dir, variant=args.variant,
                          n_episodes=args.episodes, fps=args.fps)
    if args.task in ("all", "grid"):
        print("== 2x2 grid GIF ==")
        build_grid_gif(out_dir, variant=args.variant)
    if args.task in ("all", "compare"):
        print("== 3-DQN comparison GIF ==")
        build_compare_gif(args.compare_env, out_dir,
                          n_episodes=args.episodes, fps=args.fps)

    print(f"\nDone. Output in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
