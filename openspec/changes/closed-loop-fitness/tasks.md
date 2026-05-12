## 1. Dependencies

- [x] 1.1 Add `scipy~=1.14` to `pyproject.toml [project] dependencies` and `requirements.txt`
- [x] 1.2 Run `pip install scipy` and confirm `import scipy.stats; from scipy.stats import mannwhitneyu` works
- [x] 1.3 Update `.gitignore` to exclude `reports/` (the comparison tool's output dir)

## 2. train.py optional buffer injection

- [x] 2.1 Change `train(config: TrainConfig, pre_loaded_buffer: ReplayBuffer | None = None) -> Path`
- [x] 2.2 In `train()`, after creating `agent = DQNAgent(...)`: if `pre_loaded_buffer is not None`, replace `agent.buffer` with it
- [x] 2.3 Verify backward-compat: `--reward-source env --episodes 10 --seed 42` still byte-identical to baseline first 10 (no regression to ast-buffer-manager's smoke)

## 3. Summary dataclass

- [x] 3.1 Create `hermes_dqn/training/summary.py::IterationSummary` dataclass with 12 fields from spec
- [x] 3.2 Create `ClosedLoopSummary` dataclass: `exp_name`, `condition_id`, `seed`, `n_iterations`, `iterations: list[IterationSummary]`, `total_wall_time_s`
- [x] 3.3 Add `to_dict()` / `to_json_file(path)` methods with proper enum serialization

## 4. Closed-loop engine

- [x] 4.1 Create `hermes_dqn/training/closed_loop.py::run_closed_loop(exp_name, condition_id, seed, n_iterations=5, dqn_episodes=1500, memory_db=None, decay_factor=0.5, eval_n_episodes=100, out_root="runs") -> ClosedLoopSummary`
- [x] 4.2 Resolve `memory_db` default to `<out_root>/<exp_name>/<condition_id>/memory.sqlite`
- [x] 4.3 Resolve `seed_dir = <out_root>/<exp_name>/<condition_id>/seed_<NN>/`, create
- [x] 4.4 Loop iteration 1..N:
  - 4.4.1 Build `iter_dir = seed_dir/iter_<II>/`
  - 4.4.2 Open MemoryStore; fetch top-K priors (k=5 by default, fitness_floor=-inf)
  - 4.4.3 Build TrainConfig with `out_dir=iter_dir`, `reward_source="llm"`, `memory_db=memory_db`, `memory_top_k=5`, `no_memory=False`, `seed=seed + (iter-1)*1000` (offset so each iter has distinct env reset seeds)
  - 4.4.4 If iter > 1: load `prev_iter_dir/buffer.npz` into a fresh ReplayBuffer
  - 4.4.5 Compute `diff_rewards(prev_reward_src, this_iter_reward_src)` AFTER `train()` resolves the reward (use the reward_src from llm_attempts.jsonl's last accepted entry)
    - Or simpler: do diff BEFORE invoking train() — pre-generate the reward via direct LLM call, then diff against prev_src, then pass to train via a small helper
  - 4.4.6 Apply BufferAction via `apply_policy(buf, action, decay_factor)`
  - 4.4.7 Call `train(config, pre_loaded_buffer=buf)`
  - 4.4.8 After train(): read `config.json` from iter_dir (now has env_native_mean), read fitness, build IterationSummary
  - 4.4.9 Save buffer to `iter_dir/buffer.npz` (load model + agent, get its buffer, save it). Or pass buffer through and save before train() returns
  - 4.4.10 Append IterationSummary to summary list
- [x] 4.5 On Gemma failure / train crash mid-iter: mark `status="failed"`, log error, continue to next iter
- [x] 4.6 Write `seed_dir/summary.json` at end
- [x] 4.7 Return ClosedLoopSummary
- [x] 4.8 CLI `main()`: argparse for all flags + invocation + summary print

## 5. Buffer extraction from agent

- [x] 5.1 Add `DQNAgent.get_buffer() -> ReplayBuffer` returning `self.buffer` so closed_loop can call `agent.get_buffer().save(...)` after train()
- [x] 5.2 Train.py must NOT close/destroy the agent after training so the buffer remains accessible; if currently it does, expose buffer earlier

  Implementation note: cleaner alternative is to return the agent from train() OR have closed_loop construct the agent itself and pass to train. For simplicity, give train() an optional `save_buffer_to: Path | None = None` that handles save inside train()

- [x] 5.3 Decide approach: extending train() with save_buffer_to flag is simpler. Use that.

## 6. compare_conditions.py

- [x] 6.1 Argparse: `--exp <name>`, `--conditions A,B,C` (comma-separated), `--out <reports_dir>` (default `reports/<exp>`), `--bootstrap-seed <N>` (default 12345), `--last-iter` (default True; use last iter's env_native_mean per seed)
- [x] 6.2 Discover runs: `glob(runs/<exp>/<cond>/seed_*/iter_*/config.json)`
- [x] 6.3 For each condition: collect per-seed env_native_mean (from last iter), success_rate, crash_rate, wall_time_s; flag divergent runs
- [x] 6.4 Pairwise Mann-Whitney U via `scipy.stats.mannwhitneyu(..., alternative='two-sided')`
- [x] 6.5 Bootstrap CI: 5000 resamples with fixed seed; np.percentile(2.5, 97.5)
- [x] 6.6 Win 3-condition logic + classification
- [x] 6.7 Write `reports/<exp>/comparison_report.md`:
  - Per-condition summary table (sorted by env_native_mean desc)
  - Pairwise p-value matrix
  - Win/loss/inconclusive table
  - `## Outliers` section listing divergent runs
  - `## Compute` section with total GPU-hours + hardware declaration
- [x] 6.8 Generate `training_curves.png` via matplotlib (300 dpi, x=episode, y=return per condition with shaded CI across seeds; aggregated from `iter_<last>/episodes.jsonl`)
- [x] 6.9 Generate `iteration_fitness.png` (x=iter 1..N, y=env_native_mean, one line per (cond, seed))

## 7. Pilot smoke

- [x] 7.1 Run `python -m hermes_dqn.training.closed_loop --exp-name pilot --condition-id B3-pilot --seed 42 --iterations 3 --episodes 1500` (~ 75 min wall-time)
- [x] 7.2 Verify `runs/pilot/B3-pilot/seed_42/summary.json` lists 3 iter entries with all expected fields
- [x] 7.3 Verify iter 2 / 3 prompts contain "PRIOR HIGH-FITNESS ATTEMPTS" (read iter_02/llm_attempts.jsonl[0].prompt)
- [x] 7.4 Verify `iter_02/buffer.npz` and `iter_03/buffer.npz` exist and load cleanly
- [x] 7.5 Verify diff_from_prev / buffer_action are non-null for iter 2/3 and match the AST diff classifier output
- [x] 7.6 Print env_native_mean across iter 1/2/3 to console (no statistical claim, just verify it's monotone or close)
- [x] 7.7 Run `python tools/compare_conditions.py --exp pilot --conditions B3-pilot` to verify the stats tool runs on a single-condition input (will produce a report with no comparisons but with summary stats)

## 8. Wrap-up

- [x] 8.1 `openspec validate closed-loop-fitness --strict` passes
- [x] 8.2 All scenarios across 2 spec files (closed-loop-engine 6 Requirements / condition-comparison 5 Requirements, ~20 scenarios total) verified by pilot (7.x) OR explicitly deferred ("requires full 5-seed × 6-condition run; pilot covers mechanism")
- [x] 8.3 README.md updated: bump version to 0.3.0, mark all 3 core contributions as ✅ implemented, add "## Reproducing the paper Table 1" subsection pointing to `tools/compare_conditions.py` invocation
- [x] 8.4 Ready to `/opsx:archive` once next handover is written
