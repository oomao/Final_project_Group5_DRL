# closed-loop-engine Specification

## Purpose
TBD - created by archiving change closed-loop-fitness. Update Purpose after archive.
## Requirements
### Requirement: Closed-loop iteration engine
The system SHALL provide `hermes_dqn.training.closed_loop.run_closed_loop(exp_name, condition_id, seed, n_iterations=5, dqn_episodes=1500, memory_db=None, decay_factor=0.5, eval_n_episodes=100, out_root="runs") -> ClosedLoopSummary` that runs the full Hermes-DQN 7-step loop for `n_iterations` rounds and returns a structured summary.

#### Scenario: Default pilot invocation succeeds
- **WHEN** `run_closed_loop("pilot", "B3-pilot", seed=42, n_iterations=3, dqn_episodes=1500)` is called with valid GOOGLE_API_KEY
- **THEN** the function MUST return a `ClosedLoopSummary` whose `iterations` list has 3 entries
- **AND** `runs/pilot/B3-pilot/seed_42/` MUST exist with `summary.json` and 3 sub-directories `iter_01/` through `iter_03/`
- **AND** each `iter_NN/` MUST contain `config.json`, `episodes.jsonl`, `reward_fn.py`, `model_final.pt`, `llm_attempts.jsonl`, and `buffer.npz`

#### Scenario: Run hierarchy matches experiments-protocol R6
- **WHEN** any closed-loop run completes
- **THEN** the directory layout MUST be `runs/<exp_name>/<condition_id>/seed_<NN>/iter_<II>/` exactly (zero-padded NN/II to 2 digits)
- **AND** the structure MUST satisfy `establish-project-lifecycle-spec / experiments-protocol` Requirement "Run Directory Hierarchy" for evaluation-grade runs

### Requirement: Prior reward fed via Hermes memory
Each iteration after the first SHALL call `MemoryStore.top_k_by_fitness(k=5)` against the iteration's configured `memory_db` and pass the returned entries to `LLMRewardClient.generate(memory=priors)`. The default memory_db SHALL be `runs/<exp_name>/<condition_id>/memory.sqlite` (a per-condition store).

#### Scenario: Iteration 2 sees iteration 1's reward
- **WHEN** the pilot runs `n_iterations=3, seed=42`
- **THEN** iteration 2's `llm_attempts.jsonl[0].prompt` MUST contain a "PRIOR HIGH-FITNESS ATTEMPTS" section
- **AND** that section MUST quote iteration 1's `reward_code` verbatim inside a fenced Python block

#### Scenario: Iteration 1 sees empty priors
- **WHEN** the closed loop starts iteration 1 against a fresh memory_db
- **THEN** iteration 1's prompt MUST NOT contain a "PRIOR HIGH-FITNESS ATTEMPTS" section
- **AND** `iter_01/config.json["memory_priors_used"]` MUST equal `[]`

### Requirement: Inter-iteration buffer handoff via AST policy
After iteration N completes, the agent's replay buffer SHALL be saved to `iter_<NN>/buffer.npz`. Iteration N+1 SHALL load that buffer, compute `diff_rewards(prev_reward_src, new_reward_src)`, derive the `BufferAction` via `decide_policy`, and call `apply_policy` with the configured `decay_factor` before training begins.

#### Scenario: KEEP path on numeric-only diff
- **WHEN** iteration N+1's reward differs from iteration N's only in numeric constants (NUMERIC_DIFF)
- **THEN** the inherited buffer MUST be loaded
- **AND** all sample weights MUST remain at 1.0 (no decay applied)
- **AND** `summary.json` MUST record `iterations[N].buffer_action == "keep"` and `diff_from_prev.kind == "NUMERIC_DIFF"`

#### Scenario: DECAY path on structural diff
- **WHEN** iteration N+1's reward is structurally different but similar (STRUCTURAL_DIFF, similarity > 0.7)
- **THEN** the inherited buffer MUST be loaded
- **AND** `apply_policy(buffer, BufferAction.DECAY, decay_factor)` MUST be called before training, scaling all loaded samples' weights by `decay_factor`
- **AND** `summary.json` MUST record `buffer_action == "decay"`

#### Scenario: CLEAR path on total rewrite
- **WHEN** iteration N+1's reward is wildly different (TOTAL_REWRITE, similarity <= 0.7)
- **THEN** the inherited buffer MUST be loaded then cleared via `apply_policy(buffer, BufferAction.CLEAR)`
- **AND** training MUST proceed with an empty buffer (warm-up phase repeats)
- **AND** `summary.json` MUST record `buffer_action == "clear"`

#### Scenario: First iteration uses fresh buffer
- **WHEN** iteration 1 starts
- **THEN** no buffer load MUST be attempted (no prior iteration exists)
- **AND** `summary.json` MUST record `iterations[0].diff_from_prev == null` and `buffer_action == null`

### Requirement: Iteration-level summary persistence
The function SHALL write `runs/<exp_name>/<condition_id>/seed_<NN>/summary.json` capturing per-iteration: `iter`, `reward_fn_sha256`, `memory_priors_used`, `diff_from_prev`, `buffer_action`, `env_native_mean`, `env_native_success`, `env_native_crash_rate`, `shaped_mean_last100`, `converge_episode`, `wall_time_s`, `status` (`"ok"` or `"failed"`), and a top-level `total_wall_time_s`.

#### Scenario: Summary schema is complete
- **WHEN** any closed-loop run completes (even with failures)
- **THEN** `summary.json` MUST exist with all top-level fields and one entry per attempted iteration
- **AND** each iteration entry MUST include the 12 fields above

#### Scenario: Failure does not abort subsequent iterations
- **WHEN** Gemma generation fails for iteration 2 (e.g., all 3 retries exhausted)
- **THEN** iteration 2's summary entry MUST have `status == "failed"` and a non-null `error` field
- **AND** iteration 3 MUST still attempt to run with an empty buffer and no iteration-2 prior
- **AND** iteration 3 reads priors from memory_db (which still contains iter 1's entry)

### Requirement: train.py accepts a pre-loaded replay buffer
`hermes_dqn.training.train.train(config, pre_loaded_buffer=None)` SHALL accept an optional `ReplayBuffer` instance. When provided, the function SHALL use it instead of constructing a fresh buffer inside the agent. When `None` (the default), behavior SHALL be byte-identical to the gemma-reward-generator-era `train()`.

#### Scenario: Backward-compat when pre_loaded_buffer omitted
- **WHEN** any prior caller (e.g., `python -m hermes_dqn.training.train --reward-source env --episodes 10 --seed 42`) invokes train
- **THEN** the per-episode `return` sequence MUST be byte-identical to `runs/baseline_seed42/episodes.jsonl` first 10 rows
- **AND** no new file or RNG-state difference MUST occur

#### Scenario: Pre-loaded buffer is used
- **WHEN** `train(config, pre_loaded_buffer=loaded_buf)` is called and `loaded_buf` already contains 5000 transitions
- **THEN** the DQNAgent's `_size` MUST start at 5000 (not 0)
- **AND** training MUST begin gradient updates earlier (no warm-up wait) because `len(buffer) >= train_start` already

### Requirement: CLI entry point
The module SHALL be invocable as `python -m hermes_dqn.training.closed_loop` with argparse flags: `--exp-name`, `--condition-id`, `--seed`, `--iterations` (default 5), `--episodes` (per-iter, default 1500), `--memory-db` (optional override), `--decay-factor` (default 0.5), `--out-root` (default `runs`).

#### Scenario: CLI accepts standard flags
- **WHEN** the user runs `python -m hermes_dqn.training.closed_loop --exp-name pilot --condition-id B3-pilot --seed 42 --iterations 3`
- **THEN** the script MUST run 3 iterations end-to-end
- **AND** exit code MUST be 0 even if some iterations are marked `status: failed` (failure is per-iter, not program-wide)

#### Scenario: Per-condition memory DB is the default
- **WHEN** the CLI is invoked without `--memory-db`
- **THEN** the memory db path MUST resolve to `runs/<exp_name>/<condition_id>/memory.sqlite`
- **AND** the file MUST be created if absent (no error)

