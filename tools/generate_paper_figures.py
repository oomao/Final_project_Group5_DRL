"""Generate 6 figures for the conference paper.

Outputs:
  paper/figures/fig1_architecture.png      System architecture block diagram
  paper/figures/fig2_headline.png          Cross-env Hermes-full vs B0 bars
  paper/figures/fig3_variance_signature.png Per-seed scatter per env
  paper/figures/fig4_memory_effect.png     Memory effect direction per env
  paper/figures/fig5_per_iter_trajectories.png  LL vs MC iter-by-iter
  paper/figures/fig6_cartpole_boxplot.png  CartPole all-conditions boxplot
"""

import json
import os
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

ENVS = {
    "LunarLander-v3": "runs/final",
    "CartPole-v1":    "runs/final_cp",
    "MountainCar-v0": "runs/final_mc",
    "Acrobot-v1":     "runs/final_acr",
}
SHORT = {
    "LunarLander-v3": "LL (dense)",
    "CartPole-v1":    "CP (sparse)",
    "MountainCar-v0": "MC (sparse)",
    "Acrobot-v1":     "Acr (sparse)",
}
CONDS = ["B0-env-native", "B1-handcrafted", "B2-gemma-oneshot",
         "B3-hermes-full", "B3-no-memory", "B3-no-AST"]
COND_SHORT = {
    "B0-env-native":    "B0",
    "B1-handcrafted":   "B1",
    "B2-gemma-oneshot": "B2",
    "B3-hermes-full":   "B3 full",
    "B3-no-memory":     "B3-noMem",
    "B3-no-AST":        "B3-noAST",
}


def get_means(root, cond):
    vals = []
    for s in [42, 43, 44, 45, 46]:
        d = Path(root) / cond / f"seed_{s}"
        last = sorted(d.glob("iter_*"))[-1]
        cfg = json.load(open(last / "config.json"))
        vals.append(cfg["env_native_mean"])
    return vals


# ============ Figure 1: System architecture (horizontal, DQN-centered) ============
def fig1_architecture():
    """Closed-loop architecture in compact horizontal layout (fits text width).

    Design notes:
      - figsize matches typical paper text width (~6.5 in) so LaTeX renders at
        near 1:1 scale -> text stays readable.
      - All fonts >= 11pt source. Arrow labels go ABOVE arrows (offset +0.5)
        to avoid colliding with DQN's big box.
      - Short labels: "priors", "reward.py", "buffer", "model".
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    Y_CENTER = 4.0

    SMALL_W = 1.6
    SMALL_H = 1.7
    DQN_W = 3.4
    DQN_H = 3.4

    # Centers of the 5 stages — spread further apart so arrow labels fit in gaps
    x_mem  = 1.3
    x_gem  = 4.0
    x_ast  = 6.7
    x_dqn  = 10.0
    x_eval = 13.5

    def small_box(xc, label, color, sub):
        x0 = xc - SMALL_W / 2
        y0 = Y_CENTER - SMALL_H / 2
        rect = FancyBboxPatch((x0, y0), SMALL_W, SMALL_H,
                              boxstyle="round,pad=0.08", linewidth=1.4,
                              edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(xc, Y_CENTER + 0.30, label, ha="center", va="center",
                fontsize=11, weight="bold")
        ax.text(xc, Y_CENTER - 0.35, sub, ha="center", va="center",
                fontsize=8.5, style="italic", color="#555555")

    # Support modules
    small_box(x_mem,  "Long-term\nMemory",   "#FFE4B5", "SQLite\nFTS5")
    small_box(x_gem,  "Gemma 4 31B",         "#B0E0E6", "open-source\nLLM author")
    small_box(x_ast,  "AST diff +\nBuffer",  "#C8E6C9", "KEEP/DECAY/\nCLEAR")
    small_box(x_eval, "Env-native\nEval",    "#E1BEE7", "100 unseen\nseeds")

    # ---- DQN centerpiece ----
    x0 = x_dqn - DQN_W / 2
    y0 = Y_CENTER - DQN_H / 2
    dqn_rect = FancyBboxPatch((x0, y0), DQN_W, DQN_H,
                              boxstyle="round,pad=0.15", linewidth=3.0,
                              edgecolor="#B71C1C", facecolor="#FFCDD2")
    ax.add_patch(dqn_rect)
    ax.text(x_dqn, y0 + DQN_H - 0.4,
            "DQN AGENT", ha="center", va="center",
            fontsize=14, weight="bold", color="#B71C1C")
    ax.text(x_dqn, y0 + DQN_H - 0.85,
            "the actual learner",
            ha="center", va="center", fontsize=10, style="italic", color="#555555")
    sub_lines = [
        "Q-network: 64x64 MLP",
        "Target net: lagged copy",
        "Replay buffer: 100K cap",
        "Eps-greedy: 1.0 -> 0.01",
    ]
    sub_y0 = y0 + DQN_H - 1.40
    for i, line in enumerate(sub_lines):
        ax.text(x_dqn - DQN_W / 2 + 0.25, sub_y0 - i * 0.40,
                "* " + line, ha="left", va="center",
                fontsize=10, color="#333333")
    ax.text(x_dqn, y0 + 0.30,
            "trains N episodes per iter",
            ha="center", va="center", fontsize=9, style="italic", color="#666666")

    # ---- Forward arrows + labels (labels just ABOVE arrow line, in the
    # horizontal gap between boxes — stays below box tops including DQN's) ----
    arrow_specs = [
        (x_mem + SMALL_W / 2, x_gem  - SMALL_W / 2, "(1) priors"),
        (x_gem + SMALL_W / 2, x_ast  - SMALL_W / 2, "(2) reward.py"),
        (x_ast + SMALL_W / 2, x_dqn  - DQN_W   / 2, "(3) buffer"),
        (x_dqn + DQN_W   / 2, x_eval - SMALL_W / 2, "(4) model"),
    ]
    # Place labels ABOVE the tallest box (DQN top) so they never overlap any box.
    label_y = Y_CENTER + DQN_H / 2 + 0.30
    for x_s, x_e, label in arrow_specs:
        ax.annotate("", xy=(x_e, Y_CENTER), xytext=(x_s, Y_CENTER),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="black"))
        ax.text((x_s + x_e) / 2, label_y, label,
                ha="center", va="bottom", fontsize=10, style="italic")

    # ---- Feedback arrow ----
    fb_y = 1.5
    ax.annotate("", xy=(x_eval, fb_y), xytext=(x_eval, Y_CENTER - SMALL_H / 2),
                arrowprops=dict(arrowstyle="-", lw=1.8, color="#1565C0"))
    ax.annotate("", xy=(x_mem, fb_y), xytext=(x_eval, fb_y),
                arrowprops=dict(arrowstyle="-", lw=1.8, color="#1565C0"))
    ax.annotate("", xy=(x_mem, Y_CENTER - SMALL_H / 2),
                xytext=(x_mem, fb_y),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#1565C0"))
    ax.text((x_mem + x_eval) / 2, fb_y - 0.35,
            "(5) fitness writeback - closes the loop",
            ha="center", va="top", fontsize=10, style="italic", color="#1565C0")

    # Top title
    ax.text(7, 7.4,
            "Closed-loop iteration (repeated N times per seed)",
            ha="center", va="center", fontsize=11, style="italic", color="#555555")

    plt.tight_layout()
    plt.savefig(OUT / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig1_architecture.png")


# ============ Figure 2: Headline cross-env ============
def fig2_headline():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    envs_list = list(ENVS.keys())
    hermes_means = [mean(get_means(ENVS[e], "B3-hermes-full")) for e in envs_list]
    b0_means = [mean(get_means(ENVS[e], "B0-env-native")) for e in envs_list]
    hermes_stds = [stdev(get_means(ENVS[e], "B3-hermes-full")) for e in envs_list]
    b0_stds = [stdev(get_means(ENVS[e], "B0-env-native")) for e in envs_list]

    x = np.arange(len(envs_list))
    width = 0.36
    ax.bar(x - width / 2, hermes_means, width, yerr=hermes_stds, capsize=4,
           label="B3-hermes-full", color="#3F88C5", edgecolor="black")
    ax.bar(x + width / 2, b0_means, width, yerr=b0_stds, capsize=4,
           label="B0-env-native", color="#D62828", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[e] for e in envs_list], fontsize=10)
    ax.set_ylabel("env_native_mean +/- std (n=5)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.legend(loc="upper right", fontsize=9)

    for i, (h, b) in enumerate(zip(hermes_means, b0_means)):
        diff = (h - b) / abs(b) * 100
        y_top = max(h, b) + (abs(max(h, b)) * 0.07 if max(h, b) != 0 else 8)
        ax.text(i, y_top, f"{diff:+.0f}%", ha="center", fontsize=10, weight="bold",
                color=("darkgreen" if diff > 10 else "black"))

    plt.tight_layout()
    plt.savefig(OUT / "fig2_headline.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig2_headline.png")


# ============ Figure 3: Variance signature ============
def fig3_variance():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    envs_list = list(ENVS.keys())
    colors = ["#D62828", "#F77F00", "#FCBF49", "#8AC926"]
    for i, e in enumerate(envs_list):
        vals = get_means(ENVS[e], "B3-hermes-full")
        ax.scatter([i] * len(vals), vals, color=colors[i], s=90, alpha=0.85,
                   edgecolor="black", zorder=3)
        m = mean(vals)
        s = stdev(vals)
        ax.errorbar(i, m, yerr=s, fmt="_", color="black", capsize=10, lw=2, zorder=4)
        ax.text(i + 0.18, m, f"std={s:.1f}", va="center", fontsize=9)
    ax.set_xticks(range(len(envs_list)))
    ax.set_xticklabels([SHORT[e] for e in envs_list], fontsize=10)
    ax.set_ylabel("B3-hermes-full per-seed env_native_mean")
    ax.axhline(0, color="gray", lw=0.4)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_variance_signature.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig3_variance_signature.png")


# ============ Figure 4: Memory effect direction ============
def fig4_memory_effect():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    envs_list = list(ENVS.keys())
    diffs = []
    for e in envs_list:
        h = get_means(ENVS[e], "B3-hermes-full")
        n = get_means(ENVS[e], "B3-no-memory")
        diff = (mean(h) - mean(n)) / abs(mean(n)) * 100 if abs(mean(n)) > 1e-9 else 0
        diffs.append(diff)
    labels = [SHORT[e] for e in envs_list]
    colors_bar = ["#D62828" if d < -10 else ("#8AC926" if d > 10 else "gray") for d in diffs]
    bars = ax.bar(labels, diffs, color=colors_bar, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(10, color="green", lw=0.6, ls="--", alpha=0.5,
               label="+/-10% effect threshold")
    ax.axhline(-10, color="red", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel("Memory effect = (Hermes - no-memory) / |no-memory| * 100%")
    ax.legend(loc="upper left", fontsize=9)

    # Manually set y-axis to give headroom both above (for positive labels)
    # and below (for the LL negative bar plus p-value annotations).
    y_max = max(max(diffs) + 12, 50)
    y_min = min(min(diffs) - 18, -55)
    ax.set_ylim(y_min, y_max)

    # diff% labels: positive bars get label ABOVE; negative bars get label BELOW
    for bar, d in zip(bars, diffs):
        y_lbl = d + 2 if d >= 0 else d - 4
        va = "bottom" if d >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_lbl, f"{d:+.1f}%",
                ha="center", va=va, fontsize=10, weight="bold")

    # p-value annotations: place in a dedicated band BELOW the x-axis, well below
    # any negative bar. Use figure's bottom margin.
    ps = [0.0317, 0.2222, 0.1425, 0.7533]
    pval_y = y_min + 3  # safe band: 3 units above the very bottom of axis
    for i, p in enumerate(ps):
        sig = "**" if p < 0.05 else "n.s."
        ax.text(i, pval_y, f"p={p}\n({sig})",
                ha="center", va="bottom", fontsize=8, color="#444444")

    plt.tight_layout()
    plt.savefig(OUT / "fig4_memory_effect.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig4_memory_effect.png")


# ============ Figure 5: Per-iter trajectories ============
def fig5_per_iter():
    # Stack vertically so each subplot has full width and breathing room;
    # use a shared figure to keep alignment.
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    panel_data = [("LunarLander-v3 (dense reward)", "runs/final"),
                  ("MountainCar-v0 (sparse reward)", "runs/final_mc")]
    seed_colors = {42: "#1565C0", 43: "#D62828", 44: "#388E3C",
                   45: "#6A1B9A", 46: "#EF6C00"}
    for ax, (env_label, root) in zip(axes, panel_data):
        for s in [42, 43, 44, 45, 46]:
            d = Path(root) / "B3-hermes-full" / f"seed_{s}"
            iter_vals = []
            for i in range(1, 6):
                cf = d / f"iter_{i:02d}" / "config.json"
                if cf.exists():
                    val = json.load(open(cf)).get("env_native_mean", None)
                    if val is not None:
                        iter_vals.append(val)
            xs = list(range(1, len(iter_vals) + 1))
            ax.plot(xs, iter_vals, "o-", label=f"seed {s}",
                    color=seed_colors[s], lw=1.8, markersize=7)
        ax.set_ylabel("env_native_mean (eval)")
        ax.set_title(env_label, fontsize=11, loc="left")
        ax.grid(True, alpha=0.3)
        # Legend outside top-right to avoid overlapping any seed trajectory
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=True, title="seed")
    axes[-1].set_xlabel("iteration")
    axes[-1].set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(OUT / "fig5_per_iter_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig5_per_iter_trajectories.png")


# ============ Figure 6: CartPole all-conditions boxplot ============
def fig6_cartpole_box():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cp_data = [get_means(ENVS["CartPole-v1"], c) for c in CONDS]
    bp = ax.boxplot(cp_data, labels=[COND_SHORT[c] for c in CONDS], patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    colors_bp = ["#D62828", "#F77F00", "#FCBF49", "#3F88C5", "#8AC926", "#9D4EDD"]
    for patch, c in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("env_native_mean (100 unseen eval seeds)")
    ax.axhline(475, color="gray", lw=0.6, ls="--", alpha=0.7)
    ax.text(6.4, 475, "475 = solved", fontsize=8, va="center")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_cartpole_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig6_cartpole_boxplot.png")


# ============ Part 2: Figure 7 — Hermes-vs-B0 across DQN variants ============
def fig7_part2_hermes_vs_b0():
    """Per-env, per-variant: Hermes mean vs B0 mean with WIN markers.

    Layout: 4 subplots (one per env), each shows 3 variant groups (vanilla,
    Double, Dueling), each group has 2 bars (B0 in red, Hermes in blue).
    Above each bar pair, annotate Δ% and the WIN / (n.s.) verdict.
    """
    from scipy.stats import mannwhitneyu  # local import to avoid global cost
    variants = {
        "vanilla": {"cp": "runs/final_cp",         "mc": "runs/final_mc",
                    "acr": "runs/final_acr",       "ll": "runs/final"},
        "double":  {"cp": "runs/part2_double_cp",  "mc": "runs/part2_double_mc",
                    "acr": "runs/part2_double_acr","ll": "runs/part2_double_ll"},
        "dueling": {"cp": "runs/part2_dueling_cp", "mc": "runs/part2_dueling_mc",
                    "acr": "runs/part2_dueling_acr","ll": "runs/part2_dueling_ll"},
    }
    env_short = ["cp", "mc", "acr", "ll"]
    env_titles = ["CartPole-v1 (sparse)", "MountainCar-v0 (sparse)",
                  "Acrobot-v1 (sparse)",  "LunarLander-v3 (dense)"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, env_s, env_title in zip(axes, env_short, env_titles):
        x = np.arange(3)
        width = 0.36
        b0_means, b3_means = [], []
        b0_stds,  b3_stds  = [], []
        diffs, pvals = [], []
        for v in ["vanilla", "double", "dueling"]:
            b0 = []
            b3 = []
            for s in [42, 43, 44, 45, 46]:
                d0 = Path(variants[v][env_s]) / "B0-env-native" / f"seed_{s}"
                cfg0 = json.load(open(sorted(d0.glob("iter_*"))[-1] / "config.json"))
                b0.append(cfg0["env_native_mean"])
                d3 = Path(variants[v][env_s]) / "B3-hermes-full" / f"seed_{s}"
                cfg3 = json.load(open(sorted(d3.glob("iter_*"))[-1] / "config.json"))
                b3.append(cfg3["env_native_mean"])
            b0_means.append(mean(b0))
            b3_means.append(mean(b3))
            b0_stds.append(stdev(b0))
            b3_stds.append(stdev(b3))
            u = mannwhitneyu(b3, b0, alternative="two-sided")
            pvals.append(float(u.pvalue))
            diffs.append((mean(b3) - mean(b0)) / abs(mean(b0)) * 100 if abs(mean(b0)) > 1e-9 else 0)

        ax.bar(x - width / 2, b0_means, width, yerr=b0_stds, capsize=3,
               label="B0-env-native", color="#D62828", edgecolor="black")
        ax.bar(x + width / 2, b3_means, width, yerr=b3_stds, capsize=3,
               label="B3-hermes-full", color="#3F88C5", edgecolor="black")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["van", "Double", "Dueling"], fontsize=9)
        ax.set_title(env_title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        # Per-variant verdict + Δ% above bars
        for i in range(3):
            h = max(b0_means[i] + b0_stds[i], b3_means[i] + b3_stds[i])
            l = min(b0_means[i] - b0_stds[i], b3_means[i] - b3_stds[i])
            top = h + 0.08 * (abs(h) if abs(h) > 1 else 5)
            sig = "WIN" if pvals[i] < 0.05 and abs(diffs[i]) >= 10 else "(n.s.)"
            color = "darkgreen" if sig == "WIN" else "#555555"
            ax.text(x[i], top, f"{diffs[i]:+.0f}%\n{sig}",
                    ha="center", va="bottom", fontsize=8, color=color, weight="bold")
        # Pad y-limits so labels fit
        ax_min, ax_max = ax.get_ylim()
        ax.set_ylim(ax_min, ax_max + (ax_max - ax_min) * 0.15)

    axes[0].set_ylabel("env_native_mean (n=5, +/- std)", fontsize=9)
    # Single shared legend at bottom
    axes[-1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_part2_hermes_vs_b0.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig7_part2_hermes_vs_b0.png")


# ============ Part 2: Figure 8 — Hermes robustness across DQN variants ============
def fig8_part2_hermes_robustness():
    """B3-hermes-full only, across 3 DQN variants and 4 envs.

    Per env: 3 bars (vanilla / Double / Dueling) showing B3-hermes-full mean
    with std errorbars. Pairwise p-values annotated below. Visualizes the
    "Hermes is agent-agnostic" claim: bars within each env should look
    statistically indistinguishable (matches p > 0.3 finding).
    """
    variants_paths = {
        "vanilla": {"cp": "runs/final_cp", "mc": "runs/final_mc",
                    "acr": "runs/final_acr", "ll": "runs/final"},
        "double":  {"cp": "runs/part2_double_cp", "mc": "runs/part2_double_mc",
                    "acr": "runs/part2_double_acr", "ll": "runs/part2_double_ll"},
        "dueling": {"cp": "runs/part2_dueling_cp", "mc": "runs/part2_dueling_mc",
                    "acr": "runs/part2_dueling_acr", "ll": "runs/part2_dueling_ll"},
    }
    env_short = ["cp", "mc", "acr", "ll"]
    env_titles = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "LunarLander-v3"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    colors = ["#1565C0", "#D84315", "#2E7D32"]  # vanilla / Double / Dueling
    for ax, env_s, env_title in zip(axes, env_short, env_titles):
        means, stds = [], []
        for v in ["vanilla", "double", "dueling"]:
            vals = []
            for s in [42, 43, 44, 45, 46]:
                d = Path(variants_paths[v][env_s]) / "B3-hermes-full" / f"seed_{s}"
                cfg = json.load(open(sorted(d.glob("iter_*"))[-1] / "config.json"))
                vals.append(cfg["env_native_mean"])
            means.append(mean(vals))
            stds.append(stdev(vals))
        x = np.arange(3)
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="black", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["vanilla", "Double", "Dueling"], fontsize=9)
        ax.set_title(env_title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0, color="gray", lw=0.4)
        for bar, m, s in zip(bars, means, stds):
            y_lbl = m + (s + (abs(m) * 0.05 if abs(m) > 1 else 3)) * (1 if m >= 0 else -1)
            va = "bottom" if m >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y_lbl,
                    f"{m:.1f}", ha="center", va=va, fontsize=9, weight="bold")
    axes[0].set_ylabel("B3-hermes-full env_native_mean (n=5)", fontsize=9)
    fig.suptitle("Hermes robustness across DQN variants (no pairwise difference reaches significance)",
                 fontsize=11, y=1.02, style="italic", color="#444444")
    plt.tight_layout()
    plt.savefig(OUT / "fig8_part2_hermes_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] fig8_part2_hermes_robustness.png")


if __name__ == "__main__":
    fig1_architecture()
    fig2_headline()
    fig3_variance()
    fig4_memory_effect()
    fig5_per_iter()
    fig6_cartpole_box()
    fig7_part2_hermes_vs_b0()
    fig8_part2_hermes_robustness()
    print("\n=== Generated ===")
    for f in sorted(OUT.glob("*.png")):
        sz = os.path.getsize(f) / 1024
        print(f"  {f} ({sz:.0f} KB)")
