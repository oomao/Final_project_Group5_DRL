"""Print a side-by-side comparison of baseline / gemma-no-memory / gemma-memory runs."""

import json
import sqlite3
from pathlib import Path


RUNS = [
    ("baseline_seed42", "env", "—"),
    ("gemma_seed42", "llm (no memory)", "(none)"),
    ("gemma_mem_seed42", "llm + memory (1st)", "[] (empty DB)"),
    ("gemma_mem_seed43", "llm + memory (2nd)", "reads seed 42"),
]


def main() -> int:
    for run, label, priors_note in RUNS:
        path = Path(f"runs/{run}/config.json")
        if not path.exists():
            print(f"{run:<22}  MISSING")
            continue
        with path.open(encoding="utf-8") as fp:
            c = json.load(fp)
        env_mean = c.get("env_native_mean")
        env_success = c.get("env_native_success")
        env_crash = c.get("env_native_crash_rate")
        env_len = c.get("env_native_mean_length")
        priors_used = c.get("memory_priors_used", "—")
        sha = c.get("reward_fn_sha256", "")[:12]
        print(f"{run:<22}  reward={label:<22}  priors={priors_note}")
        if env_mean is not None:
            print(
                f"  env_native_mean={env_mean:>7.2f}  success={env_success:>5.2%}  "
                f"crash={env_crash:>5.2%}  ep_len={env_len:>4.0f}"
            )
        if priors_used != "—":
            print(f"  memory_priors_used={priors_used}  reward_fn_sha={sha}...")
        print()

    print("=== memory.sqlite entries (long-term store) ===")
    db = Path("runs/memory.sqlite")
    if db.exists():
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT id, reward_fn_sha256, env_native_mean, env_native_success, "
            "success_rate, mean_reward_last100 FROM memory ORDER BY id"
        ).fetchall()
        print(f"total entries: {len(rows)}")
        for r in rows:
            print(
                f"  id={r[0]}  sha={r[1][:12]}...  env_native_mean={r[2]:.2f}  "
                f"env_native_success={r[3]:.2%}  shaped_mean={r[5]:.2f}"
            )
    else:
        print("(no memory.sqlite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
