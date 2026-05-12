"""Multi-iteration Hermes-DQN closed loop.

Per iteration the 7 steps are:

    1. Hermes memory: top_k_by_fitness(k) -> priors
    2. Gemma generate: LLMRewardClient.generate(memory=priors) -> reward source
    3. AST diff: diff_rewards(prev_src, this_src) (skipped on iter 1)
    4. Buffer policy: apply_policy(prev_buffer, decide_policy(diff), decay_factor)
    5. DQN train: train(config, pre_loaded_buffer=buf, pre_resolved_reward=(src, fn))
    6. Env-native eval is run inside train() and recorded in config.json
    7. Persist: memory.write(MemoryEntry) and buffer was already saved by train()

Library entry: ``run_closed_loop(...)`` returns a ClosedLoopSummary.
CLI entry: ``python -m hermes_dqn.training.closed_loop ...``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hermes_dqn.agent.replay_buffer import ReplayBuffer
from hermes_dqn.buffer import (
    BufferAction,
    apply_policy,
    decide_policy,
    diff_rewards,
)
from hermes_dqn.memory import MemoryEntry, MemoryStore
from hermes_dqn.training.summary import ClosedLoopSummary, IterationSummary
from hermes_dqn.training.train import TrainConfig, train


_ENV_STUB_REWARD_SRC = (
    "# env native reward (no custom function)\n"
    "def reward(obs, action, next_obs, env_reward, terminated, truncated, info):\n"
    "    return float(env_reward)\n"
)


def _generate_reward_for_iter(
    iter_dir: Path,
    memory_db: str | Path,
    memory_top_k: int,
    seed: int,
) -> tuple[str, "object", list[int]]:
    """Pull priors from memory, generate via Gemma, validate, return (src, fn, prior_ids).

    Raises ``RewardGenerationError`` if all 3 retries fail — caller decides how to
    handle (typically: mark iter as failed, continue to next).
    """
    from dotenv import load_dotenv

    from hermes_dqn.llm import LLMRewardClient, compile_reward

    load_dotenv()
    client = LLMRewardClient()

    with MemoryStore(memory_db) as store:
        priors = store.top_k_by_fitness(k=memory_top_k, fitness_floor=float("-inf"))
        prior_ids = [int(p.id) for p in priors if p.id is not None]

    attempts_log = iter_dir / "llm_attempts.jsonl"
    src = client.generate(
        attempts_log_path=attempts_log,
        memory=priors or None,
    )
    fn = compile_reward(src)
    return src, fn, prior_ids


def _make_buffer_from_prev(
    prev_iter_dir: Path,
    capacity: int,
    obs_dim: int,
    seed: int,
) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, seed=seed)
    buf.load(prev_iter_dir / "buffer.npz")
    return buf


def run_closed_loop(
    exp_name: str,
    condition_id: str,
    seed: int,
    n_iterations: int = 5,
    dqn_episodes: int = 1500,
    memory_db: str | Path | None = None,
    decay_factor: float = 0.5,
    eval_n_episodes: int = 100,
    memory_top_k: int = 5,
    out_root: str | Path = "runs",
) -> ClosedLoopSummary:
    """Run the full Hermes-DQN closed loop for one (condition, seed)."""
    seed_dir = Path(out_root) / exp_name / condition_id / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if memory_db is None:
        memory_db = Path(out_root) / exp_name / condition_id / "memory.sqlite"
    memory_db = Path(memory_db)

    summary = ClosedLoopSummary(
        exp_name=exp_name,
        condition_id=condition_id,
        seed=seed,
        n_iterations=n_iterations,
    )

    prev_reward_src: str | None = None
    prev_iter_dir: Path | None = None
    overall_start = time.time()

    for iter_idx in range(1, n_iterations + 1):
        iter_dir = seed_dir / f"iter_{iter_idx:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_start = time.time()

        # ---- Step 1+2: Hermes memory + Gemma generate ----
        try:
            reward_src, reward_fn, prior_ids = _generate_reward_for_iter(
                iter_dir=iter_dir,
                memory_db=memory_db,
                memory_top_k=memory_top_k,
                seed=seed,
            )
        except BaseException as e:
            wall = time.time() - iter_start
            print(
                f"[iter {iter_idx}/{n_iterations}] FAIL Gemma generation: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            summary.iterations.append(
                IterationSummary(
                    iter=iter_idx,
                    reward_fn_sha256="",
                    memory_priors_used=[],
                    diff_from_prev=None,
                    buffer_action=None,
                    env_native_mean=0.0,
                    env_native_success=0.0,
                    env_native_crash_rate=0.0,
                    shaped_mean_last100=0.0,
                    converge_episode=None,
                    wall_time_s=wall,
                    status="failed",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            # Reset chain so next iter starts fresh
            prev_reward_src = None
            prev_iter_dir = None
            continue

        # ---- Step 3: AST diff ----
        diff_dict: dict | None = None
        action: BufferAction | None = None
        if prev_reward_src is not None:
            diff = diff_rewards(prev_reward_src, reward_src)
            diff_dict = {
                "kind": diff.kind,
                "similarity": float(diff.similarity),
                "diff_summary": diff.diff_summary,
            }
            action = decide_policy(diff)

        # ---- Step 4: Buffer policy on inherited buffer ----
        pre_loaded_buffer: ReplayBuffer | None = None
        if prev_iter_dir is not None:
            pre_loaded_buffer = _make_buffer_from_prev(
                prev_iter_dir,
                capacity=100_000,  # MUST match DQNConfig.buffer_capacity
                obs_dim=8,
                seed=seed,
            )
            if action is not None:
                apply_policy(pre_loaded_buffer, action, decay_factor=decay_factor)

        # ---- Step 5: DQN train (env-native eval is inside train()) ----
        sha = hashlib.sha256(reward_src.encode("utf-8")).hexdigest()
        config = TrainConfig.from_overrides(
            {
                "seed": seed + (iter_idx - 1) * 1000,  # distinct env resets per iter
                "episodes": dqn_episodes,
                "out_dir": str(iter_dir),
                "reward_source": "llm",
                "memory_db": str(memory_db),
                "memory_top_k": memory_top_k,
                "no_memory": False,
                "eval_n_episodes": eval_n_episodes,
            }
        )
        config.memory_state = "hermes-sqlite-fts5"
        config.memory_priors_used = prior_ids

        try:
            train(
                config,
                pre_loaded_buffer=pre_loaded_buffer,
                pre_resolved_reward=(reward_src, reward_fn),
            )
        except BaseException as e:
            wall = time.time() - iter_start
            print(
                f"[iter {iter_idx}/{n_iterations}] FAIL during train: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            summary.iterations.append(
                IterationSummary(
                    iter=iter_idx,
                    reward_fn_sha256=sha,
                    memory_priors_used=prior_ids,
                    diff_from_prev=diff_dict,
                    buffer_action=action.value if action else None,
                    env_native_mean=0.0,
                    env_native_success=0.0,
                    env_native_crash_rate=0.0,
                    shaped_mean_last100=0.0,
                    converge_episode=None,
                    wall_time_s=wall,
                    status="failed",
                    error=f"{type(e).__name__}: {e}",
                )
            )
            prev_reward_src = None
            prev_iter_dir = None
            continue

        # ---- Step 6+7: read eval metrics + write memory ----
        with (iter_dir / "config.json").open("r", encoding="utf-8") as fp:
            iter_config = json.load(fp)
        env_native_mean = float(iter_config.get("env_native_mean", 0.0))
        env_native_success = float(iter_config.get("env_native_success", 0.0))
        env_native_crash = float(iter_config.get("env_native_crash_rate", 0.0))

        from hermes_dqn.training.fitness import FitnessEvaluator

        report = FitnessEvaluator().evaluate(iter_dir / "episodes.jsonl")

        entry = MemoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_dir=str(iter_dir),
            reward_fn_sha256=sha,
            reward_code=reward_src,
            converge_episode=report.converge_episode,
            mean_reward_last100=report.mean_reward_last100,
            success_rate=report.success_rate,
            env_native_mean=env_native_mean,
            env_native_success=env_native_success,
        )
        with MemoryStore(memory_db) as store:
            new_id = store.write(entry)

        wall = time.time() - iter_start
        summary.iterations.append(
            IterationSummary(
                iter=iter_idx,
                reward_fn_sha256=sha,
                memory_priors_used=prior_ids,
                diff_from_prev=diff_dict,
                buffer_action=action.value if action else None,
                env_native_mean=env_native_mean,
                env_native_success=env_native_success,
                env_native_crash_rate=env_native_crash,
                shaped_mean_last100=float(report.mean_reward_last100),
                converge_episode=report.converge_episode,
                wall_time_s=round(wall, 2),
            )
        )
        print(
            f"[iter {iter_idx}/{n_iterations}] OK env_native_mean={env_native_mean:.2f} "
            f"success={env_native_success:.2%} wall={wall:.1f}s memory_id={new_id} "
            f"diff={diff_dict['kind'] if diff_dict else 'first'}",
            flush=True,
        )

        prev_reward_src = reward_src
        prev_iter_dir = iter_dir

    summary.total_wall_time_s = round(time.time() - overall_start, 2)
    summary.to_json_file(seed_dir / "summary.json")
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Hermes-DQN closed loop for one (cond, seed).")
    p.add_argument("--exp-name", required=True, help="Experiment name (top-level runs/ dir).")
    p.add_argument("--condition-id", required=True, help="e.g. B3-pilot, B3-hermes-full, ...")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--episodes", type=int, default=1500, help="DQN episodes per iteration.")
    p.add_argument("--memory-db", type=str, default=None)
    p.add_argument("--decay-factor", type=float, default=0.5)
    p.add_argument("--eval-n-episodes", type=int, default=100)
    p.add_argument("--memory-top-k", type=int, default=5)
    p.add_argument("--out-root", type=str, default="runs")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    summary = run_closed_loop(
        exp_name=args.exp_name,
        condition_id=args.condition_id,
        seed=args.seed,
        n_iterations=args.iterations,
        dqn_episodes=args.episodes,
        memory_db=args.memory_db,
        decay_factor=args.decay_factor,
        eval_n_episodes=args.eval_n_episodes,
        memory_top_k=args.memory_top_k,
        out_root=args.out_root,
    )
    print(
        f"\n[OK] Closed loop complete: {len(summary.iterations)} iter(s), "
        f"total {summary.total_wall_time_s:.1f}s"
    )
    for it in summary.iterations:
        status_marker = " " if it.status == "ok" else "!"
        print(
            f"  {status_marker} iter {it.iter}: env_native_mean={it.env_native_mean:.2f} "
            f"success={it.env_native_success:.2%} buffer_action={it.buffer_action or '-'} "
            f"status={it.status}"
        )


if __name__ == "__main__":
    main()
