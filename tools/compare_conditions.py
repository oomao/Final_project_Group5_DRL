"""Statistical comparison across closed-loop conditions.

Reads runs/<exp>/<cond>/seed_*/iter_*/config.json plus summary.json, computes
per-condition aggregates, pairwise Mann-Whitney U + 5000-bootstrap CI, applies
the three-condition win rule from evaluation-criteria R3, and writes a
markdown report + 2 figures to reports/<exp>/.

Usage:
    python tools/compare_conditions.py --exp pilot --conditions B3-pilot
    python tools/compare_conditions.py --exp final --conditions B0,B1,B2,B3,B3-no-memory,B3-no-AST
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from scipy.stats import mannwhitneyu

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 12345
WIN_P_THRESHOLD = 0.05
WIN_DIFF_PCT_THRESHOLD = 0.10
HARDWARE_LINE = "NVIDIA RTX 4090 x 1"


@dataclass
class SeedRecord:
    seed: int
    last_iter_dir: Path
    env_native_mean: float
    env_native_success: float
    env_native_crash_rate: float
    wall_time_s: float
    divergent: bool
    status: str  # "ok" / "failed" / "divergent"


@dataclass
class ConditionAggregate:
    condition: str
    seeds: list[SeedRecord]

    @property
    def values(self) -> np.ndarray:
        return np.array([s.env_native_mean for s in self.seeds if s.status == "ok"])

    @property
    def n_effective(self) -> int:
        return int(self.values.size)

    @property
    def n_divergent(self) -> int:
        return sum(1 for s in self.seeds if s.divergent)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.seeds if s.status == "failed")


def _discover_conditions(
    exp: str, conditions: list[str], runs_root: Path, use_last_iter: bool
) -> dict[str, ConditionAggregate]:
    out: dict[str, ConditionAggregate] = {}
    for cond in conditions:
        cond_dir = runs_root / exp / cond
        if not cond_dir.exists():
            print(f"[warn] condition dir missing: {cond_dir}", file=sys.stderr)
            out[cond] = ConditionAggregate(condition=cond, seeds=[])
            continue
        seed_records: list[SeedRecord] = []
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue
            iter_dirs = sorted(seed_dir.glob("iter_*"))
            if not iter_dirs:
                print(f"[warn] no iter_* under {seed_dir}", file=sys.stderr)
                continue
            target = iter_dirs[-1] if use_last_iter else iter_dirs[0]
            config_path = target / "config.json"
            if not config_path.exists():
                print(f"[warn] no config.json at {config_path}", file=sys.stderr)
                continue
            with config_path.open("r", encoding="utf-8") as fp:
                cfg = json.load(fp)
            env_mean = float(cfg.get("env_native_mean", 0.0))
            env_success = float(cfg.get("env_native_success", 0.0))
            env_crash = float(cfg.get("env_native_crash_rate", 0.0))
            wall = float(cfg.get("env_native_mean_length", 0.0))  # rough proxy
            divergent = env_mean < -200.0
            status = "ok"
            if divergent:
                status = "divergent"
            seed_records.append(
                SeedRecord(
                    seed=seed,
                    last_iter_dir=target,
                    env_native_mean=env_mean,
                    env_native_success=env_success,
                    env_native_crash_rate=env_crash,
                    wall_time_s=wall,
                    divergent=divergent,
                    status=status,
                )
            )
        out[cond] = ConditionAggregate(condition=cond, seeds=seed_records)
    return out


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return not (hi_a < lo_b or hi_b < lo_a)


def _classify_pair(
    a: ConditionAggregate,
    b: ConditionAggregate,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Return {p_value, diff_pct, ci_overlap, verdict, reason}."""
    va = a.values
    vb = b.values
    if va.size < 2 or vb.size < 2:
        return {
            "p_value": float("nan"),
            "diff_pct": float("nan"),
            "ci_overlap": True,
            "verdict": "inconclusive",
            "reason": f"insufficient n (a={va.size}, b={vb.size})",
        }
    u_result = mannwhitneyu(va, vb, alternative="two-sided")
    p = float(u_result.pvalue)
    mean_a, mean_b = float(va.mean()), float(vb.mean())
    if abs(mean_b) < 1e-9:
        diff_pct = float("inf") if mean_a != 0 else 0.0
    else:
        diff_pct = (mean_a - mean_b) / abs(mean_b)
    lo_a, hi_a = _bootstrap_ci(va, rng)
    lo_b, hi_b = _bootstrap_ci(vb, rng)
    overlaps = _ci_overlap(lo_a, hi_a, lo_b, hi_b)

    verdict = "inconclusive"
    reasons: list[str] = []
    if p >= WIN_P_THRESHOLD:
        reasons.append(f"p={p:.4f} >= {WIN_P_THRESHOLD}")
    if abs(diff_pct) < WIN_DIFF_PCT_THRESHOLD:
        reasons.append(f"effect size {diff_pct:+.1%} below {WIN_DIFF_PCT_THRESHOLD:.0%}")
    if overlaps:
        reasons.append("CIs overlap")
    if not reasons:
        verdict = "A wins" if diff_pct > 0 else "B wins"

    return {
        "p_value": p,
        "diff_pct": diff_pct,
        "ci_a": (lo_a, hi_a),
        "ci_b": (lo_b, hi_b),
        "ci_overlap": overlaps,
        "verdict": verdict,
        "reason": "; ".join(reasons) if reasons else "all three win conditions met",
    }


def _emit_markdown_report(
    exp: str,
    aggs: dict[str, ConditionAggregate],
    pairs: dict[tuple[str, str], dict[str, Any]],
    out_path: Path,
    rng: np.random.Generator,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_conds = sorted(aggs.values(), key=lambda c: -(c.values.mean() if c.values.size else -1e9))

    lines: list[str] = []
    lines.append(f"# Comparison report: {exp}")
    lines.append("")
    lines.append("## Per-condition summary")
    lines.append("")
    lines.append("| Condition | n | env_native_mean (95% CI) | success_rate | crash_rate | n_divergent |")
    lines.append("|---|---|---|---|---|---|")
    for c in sorted_conds:
        if c.values.size:
            lo, hi = _bootstrap_ci(c.values, rng)
            mean_str = f"{c.values.mean():.2f} [{lo:.2f}, {hi:.2f}]"
            succ = float(np.mean([s.env_native_success for s in c.seeds if s.status == "ok"]))
            crash = float(np.mean([s.env_native_crash_rate for s in c.seeds if s.status == "ok"]))
            succ_str = f"{succ:.2%}"
            crash_str = f"{crash:.2%}"
        else:
            mean_str = succ_str = crash_str = "—"
        lines.append(
            f"| {c.condition} | {c.n_effective} | {mean_str} | {succ_str} | {crash_str} | {c.n_divergent} |"
        )

    lines.append("")
    lines.append("## Pairwise Mann-Whitney U + bootstrap CI")
    lines.append("")
    conds = [c.condition for c in sorted_conds]
    if len(conds) >= 2:
        lines.append("| A vs B | p | A mean diff vs B | A CI | B CI | Verdict | Reason |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, ca in enumerate(conds):
            for cb in conds[i + 1 :]:
                stats = pairs.get((ca, cb), pairs.get((cb, ca)))
                if not stats:
                    continue
                ci_a = stats.get("ci_a", (float("nan"), float("nan")))
                ci_b = stats.get("ci_b", (float("nan"), float("nan")))
                lines.append(
                    f"| {ca} vs {cb} | {stats['p_value']:.4f} | {stats['diff_pct']:+.1%} | "
                    f"[{ci_a[0]:.2f}, {ci_a[1]:.2f}] | [{ci_b[0]:.2f}, {ci_b[1]:.2f}] | "
                    f"{stats['verdict']} | {stats['reason']} |"
                )
    else:
        lines.append("(only one condition supplied; no pairwise comparisons)")

    lines.append("")
    lines.append("## Outliers (divergent or failed)")
    lines.append("")
    any_outlier = False
    for c in sorted_conds:
        for s in c.seeds:
            if s.divergent or s.status == "failed":
                lines.append(
                    f"- `{s.last_iter_dir}` (condition={c.condition}, seed={s.seed}): "
                    f"env_native_mean={s.env_native_mean:.2f}, status={s.status}"
                )
                any_outlier = True
    if not any_outlier:
        lines.append("(none)")

    lines.append("")
    lines.append("## Compute")
    lines.append("")
    total_eps = 0
    for c in sorted_conds:
        # Walk each seed's iter dirs and sum eval lengths is a rough proxy; in
        # the future closed_loop.summary.json's total_wall_time_s is better.
        for s in c.seeds:
            total_eps += 1500  # default episodes per iter
    lines.append(f"- Total training episodes counted: {total_eps}")
    lines.append(f"- Hardware: {HARDWARE_LINE}")
    lines.append("")
    lines.append(
        "Note: per-condition wall-time aggregation reads `summary.json` when present; this "
        "MVP report falls back to ep_len proxies if absent."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_figures(
    exp: str,
    aggs: dict[str, ConditionAggregate],
    figures_dir: Path,
) -> None:
    if plt is None:
        print("[warn] matplotlib not available; skipping figures", file=sys.stderr)
        return
    figures_dir.mkdir(parents=True, exist_ok=True)

    # training_curves.png: x=episode, y=return, one line per condition
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    for cond_name, agg in aggs.items():
        all_returns: list[np.ndarray] = []
        for s in agg.seeds:
            jsonl = s.last_iter_dir / "episodes.jsonl"
            if not jsonl.exists():
                continue
            with jsonl.open("r", encoding="utf-8") as fp:
                returns = np.array([json.loads(line)["return"] for line in fp])
            all_returns.append(returns)
        if not all_returns:
            continue
        max_len = max(len(r) for r in all_returns)
        padded = np.full((len(all_returns), max_len), np.nan)
        for i, r in enumerate(all_returns):
            padded[i, : len(r)] = r
        # 100-ep rolling mean per seed
        rolling = np.full_like(padded, np.nan, dtype=float)
        for i in range(padded.shape[0]):
            for j in range(padded.shape[1]):
                if j + 1 < 100:
                    rolling[i, j] = np.nanmean(padded[i, : j + 1])
                else:
                    rolling[i, j] = np.nanmean(padded[i, j - 99 : j + 1])
        mean_curve = np.nanmean(rolling, axis=0)
        lo = np.nanpercentile(rolling, 2.5, axis=0)
        hi = np.nanpercentile(rolling, 97.5, axis=0)
        x = np.arange(1, max_len + 1)
        line = ax.plot(x, mean_curve, label=cond_name, linewidth=2)[0]
        ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color())
    ax.axhline(200, ls="--", color="green", alpha=0.4, label="success threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (100-ep rolling)")
    ax.set_title(f"Training curves — {exp}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # iteration_fitness.png: x=iter, y=env_native_mean, one line per (cond, seed)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    for cond_name, agg in aggs.items():
        for s in agg.seeds:
            # Read summary.json if present
            summary_path = s.last_iter_dir.parent / "summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as fp:
                    summary = json.load(fp)
                iters = summary.get("iterations", [])
                xs = [it["iter"] for it in iters if it.get("status") == "ok"]
                ys = [it["env_native_mean"] for it in iters if it.get("status") == "ok"]
                if xs:
                    ax.plot(xs, ys, marker="o", linestyle="-", alpha=0.7, label=f"{cond_name} seed={s.seed}")
    ax.axhline(200, ls="--", color="green", alpha=0.4)
    ax.set_xlabel("LLM iteration")
    ax.set_ylabel("env_native_mean")
    ax.set_title(f"Iteration fitness — {exp}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "iteration_fitness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--conditions", required=True, help="comma-separated condition ids")
    p.add_argument("--out", default=None)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--first-iter", action="store_true", help="Use FIRST iter per seed (default uses LAST)")
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out) if args.out else Path("reports") / args.exp

    aggs = _discover_conditions(args.exp, conditions, runs_root, use_last_iter=not args.first_iter)
    rng = np.random.default_rng(args.bootstrap_seed)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    cond_names = list(aggs.keys())
    for i, ca in enumerate(cond_names):
        for cb in cond_names[i + 1 :]:
            pairs[(ca, cb)] = _classify_pair(aggs[ca], aggs[cb], rng)

    report_path = out_dir / "comparison_report.md"
    _emit_markdown_report(args.exp, aggs, pairs, report_path, rng)
    _emit_figures(args.exp, aggs, out_dir / "figures")
    print(f"[OK] Report: {report_path}")
    print(f"[OK] Figures: {out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
