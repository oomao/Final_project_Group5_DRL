"""Orchestrator for the 6-condition x 5-seed Hermes-DQN final evaluation.

Implements `experiments-protocol` (5 seeds, fixed 1500 ep, no early stop) and
`evaluation-criteria` (Complete Baseline Set: B0, B1, B2, B3, B3-no-memory,
B3-no-AST).

Design:
    * One subprocess per (condition, seed). Sequential — single GPU.
    * Idempotent: if a (cond, seed) already has a config.json with
      ``env_native_mean`` recorded, skip it. Lets you Ctrl-C and resume.
    * Failure-tolerant: a single (cond, seed) crashing logs the error and
      moves on to the next pair. Final summary lists which pairs failed.
    * Each job's stdout/stderr is teed to a per-job log file under
      ``runs/<exp>/<cond>/seed_<NN>/job.log`` for after-the-fact debugging.

Usage:
    # Smoke (10 ep, 1 iter, 1 seed, all 6 conditions, ~15 min on 4090):
    python scripts/run_full_experiment.py --exp smoke6 --episodes 10 \
        --iterations 1 --seeds 42

    # Full evaluation (1500 ep, full iter counts, 5 seeds, ~35-60 GPU-hr):
    python scripts/run_full_experiment.py --exp final --episodes 1500

    # Resume after a kill:
    python scripts/run_full_experiment.py --exp final --episodes 1500
    # (completed (cond, seed) pairs are skipped automatically)

After the run completes:
    python tools/compare_conditions.py --exp final \
        --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,\
B3-hermes-full,B3-no-memory,B3-no-AST
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `from hermes_dqn ...` imports when invoked as `python scripts/run_full_experiment.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _condition_specs(env_b1_reward_path: Path) -> dict[str, dict]:
    """Build the condition spec table. B1's reward file path depends on env."""
    return {
        "B0-env-native": {
            "kind": "train",
            "args": ["--reward-source", "env"],
            "default_iterations": 1,
        },
        "B1-handcrafted": {
            "kind": "train",
            "args": ["--reward-source", "file", "--reward-file", str(env_b1_reward_path)],
            "default_iterations": 1,
        },
        "B2-gemma-oneshot": {
            "kind": "closed_loop",
            "args": [],  # n_iter=1 + fresh memory.sqlite per seed -> empty priors
            "default_iterations": 1,
        },
        "B3-hermes-full": {
            "kind": "closed_loop",
            "args": [],
            "default_iterations": 5,
        },
        "B3-no-memory": {
            "kind": "closed_loop",
            "args": ["--no-memory"],
            "default_iterations": 5,
        },
        "B3-no-AST": {
            "kind": "closed_loop",
            "args": ["--no-ast"],
            "default_iterations": 5,
        },
    }


# Backward-compat: LunarLander conditions table for code that imports CONDITIONS directly.
CONDITIONS: dict[str, dict] = _condition_specs(
    REPO_ROOT / "experiments" / "baselines" / "B1_handcrafted.py"
)


def _seed_dir(out_root: Path, exp: str, cond: str, seed: int) -> Path:
    return out_root / exp / cond / f"seed_{seed:02d}"


def _job_already_done(seed_dir: Path, cond_spec: dict, n_iterations_override: int | None) -> bool:
    """A (cond, seed) is done iff every expected iter dir has env_native_mean
    written to config.json.
    """
    if cond_spec["kind"] == "train":
        # train.py writes directly to iter_01
        iter_dirs = [seed_dir / "iter_01"]
    else:
        n_iter = n_iterations_override or cond_spec["default_iterations"]
        iter_dirs = [seed_dir / f"iter_{i:02d}" for i in range(1, n_iter + 1)]
    for d in iter_dirs:
        cfg = d / "config.json"
        if not cfg.exists():
            return False
        try:
            with cfg.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            return False
        if "env_native_mean" not in data:
            return False
    return True


def _build_command(
    cond: str,
    cond_spec: dict,
    seed: int,
    episodes: int,
    n_iterations: int,
    seed_dir: Path,
    exp: str,
    out_root: Path,
    env_id: str = "LunarLander-v3",
) -> list[str]:
    python = sys.executable
    if cond_spec["kind"] == "train":
        # B0 / B1: single run, write directly into iter_01 to match hierarchy.
        iter_dir = seed_dir / "iter_01"
        iter_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python,
            "-m",
            "hermes_dqn.training.train",
            "--seed",
            str(seed),
            "--episodes",
            str(episodes),
            "--out-dir",
            str(iter_dir),
            "--env-id",
            env_id,
            *cond_spec["args"],
        ]
        return cmd

    # closed_loop.py path. It manages its own iter_NN sub-dirs.
    # Per-seed memory.sqlite: seeds are statistically independent (required for
    # Mann-Whitney U). Memory accumulation is only tested WITHIN a seed's 5 iters.
    per_seed_memory_db = seed_dir / "memory.sqlite"
    cmd = [
        python,
        "-m",
        "hermes_dqn.training.closed_loop",
        "--exp-name",
        exp,
        "--condition-id",
        cond,
        "--seed",
        str(seed),
        "--iterations",
        str(n_iterations),
        "--episodes",
        str(episodes),
        "--out-root",
        str(out_root),
        "--memory-db",
        str(per_seed_memory_db),
        "--env-id",
        env_id,
        *cond_spec["args"],
    ]
    return cmd


def _build_subprocess_env(cpu_threads_per_worker: int) -> dict:
    """Inherit current env + cap CPU threads so N parallel workers don't oversubscribe.

    PyTorch / NumPy honor OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS.
    With 5 workers on a 28-core CPU, 5 threads each = 25 threads total + headroom.
    """
    env = os.environ.copy()
    s = str(cpu_threads_per_worker)
    env["OMP_NUM_THREADS"] = s
    env["MKL_NUM_THREADS"] = s
    env["OPENBLAS_NUM_THREADS"] = s
    env["NUMEXPR_NUM_THREADS"] = s
    # Force unbuffered subprocess stdout so per-line logging works even when
    # we redirect straight to a file (parallel mode).
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_job(
    cmd: list[str],
    log_path: Path,
    cpu_threads_per_worker: int,
    tee_to_stdout: bool,
) -> int:
    """Run a (cond, seed) subprocess. Return its exit code.

    ``tee_to_stdout=True`` (serial mode): output goes to both terminal and log
    ``tee_to_stdout=False`` (parallel mode): output goes ONLY to log file —
    necessary because tqdm bars from 5+ parallel subprocesses would be unreadable.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sub_env = _build_subprocess_env(cpu_threads_per_worker)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# command: {' '.join(cmd)}\n")
        lf.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"# OMP_NUM_THREADS={sub_env['OMP_NUM_THREADS']}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            bufsize=1,
            text=True,
            env=sub_env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if tee_to_stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            lf.write(line)
            lf.flush()
        proc.wait()
        lf.write(f"\n# finished: {time.strftime('%Y-%m-%d %H:%M:%S')} rc={proc.returncode}\n")
        return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Experiment name. Output goes to runs/<exp>/<cond>/seed_NN/.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(CONDITIONS.keys()),
        help=f"Comma-separated condition ids. Default: all 6. Choices: {list(CONDITIONS.keys())}",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seed integers. Default: 42,43,44,45,46.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1500,
        help="DQN episodes per iteration. Default: 1500 (full run).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override n_iterations for closed-loop conditions. Default: use condition's default "
        "(1 for B2, 5 for B3*). Train conditions ignore this.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="runs",
        help="Root directory for runs. Default: runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without executing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel (cond, seed) jobs. Default 1 (serial). "
        "Setting >1 enables GPU sharing across processes; "
        "recommended max 5 for an RTX 4090 with 24GB VRAM.",
    )
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=None,
        help="OMP/MKL thread cap per subprocess. Default auto: floor(cpu_count / workers). "
        "Caps total CPU threads to roughly cpu_count to prevent oversubscription.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Seconds between parallel-mode progress reports. Default 60.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="LunarLander-v3",
        help="Gym env id. Default LunarLander-v3. Currently also supports CartPole-v1.",
    )
    args = parser.parse_args()

    # Resolve env profile (B1 reward path + default episodes).
    from hermes_dqn.env.profiles import get_profile

    try:
        profile = get_profile(args.env_id)
    except ValueError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(2)
    env_conditions = _condition_specs(REPO_ROOT / profile.b1_reward_file)
    print(
        f"[plan] env={args.env_id} (obs_dim={profile.obs_dim}, "
        f"success>={profile.success_threshold}, default_ep={profile.default_episodes})"
    )

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            print(f"[FAIL] Unknown condition: {c!r}. Known: {list(CONDITIONS.keys())}", file=sys.stderr)
            sys.exit(2)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_root = Path(args.out_root)

    # ---- Preflight: any closed_loop condition needs GOOGLE_API_KEY ----
    needs_gemma = any(env_conditions[c]["kind"] == "closed_loop" for c in conditions)
    if needs_gemma:
        # dotenv is what closed_loop.py uses; load same way for consistency
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        import os as _os

        if not _os.environ.get("GOOGLE_API_KEY"):
            print(
                "[FAIL] GOOGLE_API_KEY not set. Required for conditions: "
                f"{[c for c in conditions if env_conditions[c]['kind'] == 'closed_loop']}. "
                "Copy .env.example to .env and set GOOGLE_API_KEY=... (note: NOT 'GEMINI_API_KEY').",
                file=sys.stderr,
            )
            sys.exit(3)
        print(f"[preflight] GOOGLE_API_KEY found (len={len(_os.environ['GOOGLE_API_KEY'])})")

    # ---- Preflight: GPU check ----
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[preflight] CUDA OK: {gpu_name}")
        else:
            print("[preflight] WARNING: CUDA NOT available; training will run on CPU (very slow).")
    except ImportError:
        print("[preflight] WARNING: torch not importable from here; skip GPU check.")

    # Plan
    jobs: list[tuple[str, int, dict, int]] = []
    for cond in conditions:
        cond_spec = env_conditions[cond]
        n_iter = (
            args.iterations
            if args.iterations is not None and cond_spec["kind"] == "closed_loop"
            else cond_spec["default_iterations"]
        )
        for seed in seeds:
            jobs.append((cond, seed, cond_spec, n_iter))

    print(f"[plan] exp={args.exp} conditions={conditions} seeds={seeds} episodes={args.episodes}")
    print(f"[plan] Total (cond, seed) pairs to consider: {len(jobs)}")

    if args.dry_run:
        for cond, seed, cond_spec, n_iter in jobs:
            seed_dir = _seed_dir(out_root, args.exp, cond, seed)
            cmd = _build_command(
                cond, cond_spec, seed, args.episodes, n_iter, seed_dir, args.exp, out_root,
                env_id=args.env_id,
            )
            done = _job_already_done(seed_dir, cond_spec, n_iter)
            tag = "[SKIP]" if done else "[QUEUE]"
            print(f"  {tag} {cond} seed={seed} iter={n_iter}: {' '.join(cmd)}")
        return

    # ---- Resolve CPU thread cap ----
    workers = max(1, args.workers)
    cpu_count = os.cpu_count() or 8
    if args.cpu_threads_per_worker is not None:
        cpu_threads = args.cpu_threads_per_worker
    else:
        cpu_threads = max(1, cpu_count // workers)
    print(
        f"[plan] workers={workers}, CPU threads per worker={cpu_threads} "
        f"(cpu_count={cpu_count}, total threads={workers * cpu_threads})"
    )
    if workers > 1 and workers * cpu_threads > cpu_count:
        print(
            f"[plan] WARN: {workers} workers x {cpu_threads} threads exceeds {cpu_count} CPU cores; "
            "expect contention."
        )

    # ---- Partition jobs: already-done vs pending ----
    pending: list[tuple[str, int, dict, int]] = []
    pre_skipped = 0
    for cond, seed, cond_spec, n_iter in jobs:
        seed_dir = _seed_dir(out_root, args.exp, cond, seed)
        if _job_already_done(seed_dir, cond_spec, n_iter):
            pre_skipped += 1
            print(f"[skip] {cond} seed={seed} already complete")
        else:
            pending.append((cond, seed, cond_spec, n_iter))
    print(f"[plan] {pre_skipped} pre-skipped, {len(pending)} to run")

    results: list[tuple[str, int, str, float]] = []  # (cond, seed, status, wall_s)
    results_lock = threading.Lock()
    in_flight: set[tuple[str, int]] = set()
    in_flight_lock = threading.Lock()
    overall_start = time.time()

    def _execute_one(idx_pair: tuple[int, tuple[str, int, dict, int]]) -> None:
        idx, (cond, seed, cond_spec, n_iter) = idx_pair
        seed_dir = _seed_dir(out_root, args.exp, cond, seed)
        cmd = _build_command(
            cond, cond_spec, seed, args.episodes, n_iter, seed_dir, args.exp, out_root,
            env_id=args.env_id,
        )
        log_path = seed_dir / "job.log"
        with in_flight_lock:
            in_flight.add((cond, seed))
        if workers == 1:
            # Serial mode: keep the rich progress printout
            print(
                f"\n[{idx}/{len(pending)}] RUN {cond} seed={seed} iter={n_iter} ep={args.episodes}"
                f"\n        cmd: {' '.join(cmd)}"
                f"\n        log: {log_path}"
            )
        else:
            print(f"[{idx}/{len(pending)}] START {cond} seed={seed} (log: {log_path})", flush=True)
        job_start = time.time()
        try:
            rc = _run_job(
                cmd,
                log_path,
                cpu_threads_per_worker=cpu_threads,
                tee_to_stdout=(workers == 1),
            )
        except BaseException as e:
            wall = time.time() - job_start
            with results_lock:
                results.append((cond, seed, f"exception:{type(e).__name__}", wall))
            with in_flight_lock:
                in_flight.discard((cond, seed))
            raise
        wall = time.time() - job_start
        status = "ok" if rc == 0 else f"fail_rc{rc}"
        with results_lock:
            results.append((cond, seed, status, wall))
        with in_flight_lock:
            in_flight.discard((cond, seed))
        print(
            f"[{idx}/{len(pending)}] {status.upper()} {cond} seed={seed} in {wall/60:.1f} min",
            flush=True,
        )

    # ---- Periodic progress reporter (parallel mode only) ----
    stop_progress = threading.Event()

    def _progress_loop():
        while not stop_progress.wait(args.progress_interval):
            with results_lock:
                n_done = len(results)
            with in_flight_lock:
                running = sorted(in_flight)
            elapsed_h = (time.time() - overall_start) / 3600
            running_str = ", ".join(f"{c}/seed_{s}" for c, s in running) or "-"
            print(
                f"[progress] {n_done}/{len(pending)} done | "
                f"{len(running)} in-flight: {running_str} | "
                f"elapsed {elapsed_h:.2f}h",
                flush=True,
            )

    progress_thread: threading.Thread | None = None
    if workers > 1:
        progress_thread = threading.Thread(target=_progress_loop, daemon=True)
        progress_thread.start()

    # ---- Run ----
    try:
        if workers == 1:
            for idx, job in enumerate(pending, start=1):
                _execute_one((idx, job))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futs = {
                    executor.submit(_execute_one, (idx, job)): job
                    for idx, job in enumerate(pending, start=1)
                }
                for fut in as_completed(futs):
                    # propagate exceptions (already logged in _execute_one)
                    exc = fut.exception()
                    if exc is not None:
                        cond, seed, _, _ = futs[fut]
                        print(
                            f"[error] {cond} seed={seed} raised: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
    except KeyboardInterrupt:
        stop_progress.set()
        print(
            "\n[abort] KeyboardInterrupt; partial results remain on disk. "
            "Re-run the same command to resume — completed pairs are skipped.",
            flush=True,
        )
        sys.exit(130)
    finally:
        stop_progress.set()
        if progress_thread is not None:
            progress_thread.join(timeout=2)

    # ---- Summary ----
    total_wall = time.time() - overall_start
    n_ok = sum(1 for _, _, s, _ in results if s == "ok")
    n_fail = sum(1 for _, _, s, _ in results if s.startswith("fail") or s.startswith("exception"))
    print("\n" + "=" * 70)
    print(
        f"[done] pre-skipped {pre_skipped} + ran {len(results)} "
        f"({n_ok} ok / {n_fail} failed); total wall {total_wall/3600:.2f}h"
    )
    if n_fail > 0:
        print("\nFailed jobs:")
        for cond, seed, status, wall in results:
            if status.startswith("fail") or status.startswith("exception"):
                print(f"  - {cond} seed={seed}: {status} (wall {wall/60:.1f} min)")
        print("\nResume by re-running the same command; completed pairs will be skipped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
