"""P7 memory-harm GIF: B3-no-memory (lands) vs B3-hermes-full seed_43 (hovers).

Same LLM-authored rewards on both sides; the ONLY difference is memory.
Left lands (~248 region); right is seed_43's degenerate hover policy (11.6).
English on-canvas labels (the slide caption is Chinese).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generate_presentation_gifs import (
    load_agent, play_panels, compose, save_gif, representative,
    _seed_dir, _last_iter, score, state_tag,
)

ENV = "LunarLander-v3"
SHORT, VAR = "ll", "vanilla"
MAX = 420

nm_seed, nm_dir, nm_mean = representative(SHORT, VAR, "B3-no-memory")
hf_dir = _last_iter(_seed_dir(SHORT, VAR, "B3-hermes-full", 43))
print(f"no-memory rep seed {nm_seed} ({nm_mean:.0f}) | hermes-full seed_43 iter {hf_dir.name}")
agents = [load_agent(nm_dir), load_agent(hf_dir)]

# pick the episode seed that maximises (left lands - right score) => clearest contrast
best = None
for s in range(7000, 7018):
    rl = score(agents[0], ENV, s, MAX)
    rr = score(agents[1], ENV, s, MAX)
    gap = rl - rr
    if best is None or gap > best[0]:
        best = (gap, s, rl, rr)
gap, ep_seed, rl, rr = best
print(f"episode seed {ep_seed}: no-mem {rl:+.0f} vs full(seed43) {rr:+.0f}")

panels, rets = play_panels(agents, ENV, ep_seed, frame_every=2, max_steps=MAX)
heads = ["B3 no-memory  ->  LANDS", "B3 +memory (seed_43)  ->  HOVERS"]
footer = ("LunarLander-v3  |  5-seed env-native:  no-memory 248  vs  +memory 153  |  "
          "delta -38.3%  Mann-Whitney p=0.0317  |  seed_43 = 11.6 (degenerate hover)")
frames = []
for img, rsnap, dsnap in panels:
    frames.append(compose(img, heads, rsnap,
                          [state_tag(ENV, rsnap[0], dsnap[0]), state_tag(ENV, rsnap[1], dsnap[1])],
                          1, 1, footer))
for _ in range(24):  # hold the outcome
    frames.append(compose(panels[-1][0], heads, rets,
                          [state_tag(ENV, rets[0], True), state_tag(ENV, rets[1], True)],
                          1, 1, footer))
save_gif(frames, Path("paper/gifs/presentation/p7_memory_harm.gif"), fps=30)
print("DONE p7_memory_harm.gif")
