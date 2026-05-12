# experiments-protocol Specification

## Purpose
TBD - created by archiving change establish-project-lifecycle-spec. Update Purpose after archive.
## Requirements
### Requirement: Five Seeds per Condition
Every experimental condition SHALL be evaluated with 5 independent random seeds. The default seed list is `[42, 43, 44, 45, 46]`. n=5 with the Mann-Whitney U test gives ~60% power for medium effect sizes at α=0.05, which is the balance point between statistical defensibility and the shared-4090 throughput budget.

#### Scenario: Default seed set is launched
- **WHEN** a user starts a full evaluation of a condition
- **THEN** the system MUST produce sub-directories `seed_42/` through `seed_46/`
- **AND** each MUST contain its own `config.json`, `episodes.jsonl`, `reward_fn.py`, and `model_final.pt`

#### Scenario: Seed crash retry policy
- **WHEN** a single seed crashes for a reason that is NOT a reward-function logic error (e.g. CUDA OOM, transient I/O)
- **THEN** the system SHALL retry that seed at most once
- **AND** if it still fails, `condition_summary.json` MUST record `seed_status: "failed"` for that seed

#### Scenario: Reward-function logic error does not retry
- **WHEN** the reward function raises a Python exception during a seed's training
- **THEN** the system MUST mark the seed as `seed_status: "reward_error"` and SHALL NOT retry

### Requirement: Condition Triple Definition
A condition SHALL be uniquely identified by the triple `(reward_fn_source, memory_state, buffer_policy)`. Any change in any component yields a new condition that MUST be re-run with the full 5 seeds.

#### Scenario: Memory state toggled
- **WHEN** a developer changes `memory_state` from `"hermes-4-layer"` to `"none"` and keeps the other two components fixed
- **THEN** the new combination is a distinct condition
- **AND** MUST be re-evaluated with all 5 seeds

#### Scenario: Reward function source change
- **WHEN** the reward function source SHA-256 differs between two runs even if author claims "same logic"
- **THEN** the runs MUST be treated as distinct conditions

### Requirement: Fixed Training Budget, No Early Stop
Each run SHALL train for exactly 1500 episodes. Early stopping (terminating before episode 1500 because convergence was reached) is FORBIDDEN. A wall-time exceeding 45 minutes per seed MUST be flagged as anomalous in the run's `config.json`.

#### Scenario: Continue past convergence threshold
- **WHEN** a run reaches `mean_reward_last100 ≥ 230` at episode 800
- **THEN** training MUST continue to episode 1500
- **AND** the additional 700 episodes' data MUST be logged normally

#### Scenario: Anomalous wall-time
- **WHEN** a seed's wall-time exceeds 45 minutes on the 4090
- **THEN** `config.json` MUST record `wall_time_anomaly: true`
- **AND** the run is still included in statistics unless other failure modes apply

### Requirement: Hermes Outer Loop Fixed at 5 Iterations
For experiments involving the Hermes reward-generation loop, each seed SHALL run exactly 5 LLM iterations. Each iteration produces 1 reward-function candidate and runs the full 1500-episode training to score it.

#### Scenario: All 5 iterations succeed
- **WHEN** all 5 LLM-generated reward candidates compile and complete training
- **THEN** the seed's directory MUST contain 5 sub-iteration logs and 5 fitness reports

#### Scenario: Reward candidate fails to compile
- **WHEN** the LLM produces a reward function that fails to compile or import
- **THEN** the iteration is marked `compile_error: true`, fitness is recorded as `null`
- **AND** the system retries with a new LLM sample up to 3 times before abandoning the iteration

### Requirement: Reward Function Artifact and Integrity
Every run directory MUST contain a `reward_fn.py` file with the exact reward-function source code used for that run. `config.json` MUST record the SHA-256 hash of that file as `reward_fn_sha256`. Evaluation scripts MUST verify the hash before including a run in statistics.

#### Scenario: Reward function is persisted
- **WHEN** a run completes
- **THEN** `reward_fn.py` MUST exist in the run directory
- **AND** `config.json["reward_fn_sha256"]` MUST equal the SHA-256 of that file

#### Scenario: Hash mismatch rejection
- **WHEN** an evaluation script loads a run and the computed SHA-256 of `reward_fn.py` does not match `config.json["reward_fn_sha256"]`
- **THEN** the run MUST be excluded from statistics
- **AND** a `tampered_or_corrupt` warning MUST be printed

### Requirement: Run Directory Hierarchy
Evaluation-grade runs (1500 ep, intended for statistical inclusion) MUST use the three-level hierarchy `runs/<experiment_name>/<condition_id>/seed_<NN>/`. Ad-hoc development runs (≤ 300 ep) MAY use the legacy flat layout `runs/<YYYY-MM-DD_HH-MM-SS>/` inherited from `bootstrap-dqn-baseline`.

#### Scenario: Cross-seed aggregation glob
- **WHEN** an evaluation script needs all seeds of one condition
- **THEN** `glob("runs/<exp>/<condition>/seed_*/episodes.jsonl")` MUST return exactly 5 matching files for a complete condition

#### Scenario: Dev run keeps legacy layout
- **WHEN** a developer launches a 50-episode smoke run from their laptop
- **THEN** `runs/<timestamp>/` is acceptable and MUST NOT be rejected by tooling

#### Scenario: Mixed-mode coexistence
- **WHEN** the `runs/` directory contains both a flat-timestamp directory and a 3-level hierarchy
- **THEN** evaluation tools MUST process only the 3-level entries and ignore flat-timestamp ones

### Requirement: Three Training Sizes
The system SHALL support three named training-run sizes: `smoke` (10 episodes, CI / wiring check), `pilot` (300 episodes, sanity / effect-size estimation), and `full` (1500 episodes, the only size eligible for statistical inclusion).

#### Scenario: Smoke run excluded from statistics
- **WHEN** an evaluation script encounters a run with `total_episodes == 10`
- **THEN** the run MUST be tagged `excluded: smoke`
- **AND** MUST NOT contribute to any reported metric

#### Scenario: Pilot run excluded from final statistics
- **WHEN** a run has `total_episodes == 300`
- **THEN** it is tagged `excluded: pilot`
- **AND** MAY be cited in `design.md` as effect-size estimation evidence but MUST NOT appear in final tables

#### Scenario: Full run is eligible
- **WHEN** a run has `total_episodes == 1500` and passes all other gates
- **THEN** it is eligible for inclusion in the final statistics table

### Requirement: GPU Claim Protocol
Before launching any `full`-size run on the shared 4090, the developer MUST execute `scripts/claim-gpu.sh`, which (a) runs `nvidia-smi`, (b) aborts if GPU 0 utilisation > 50% or if memory used > 1 GB by another user, and (c) prints the current `docs/4090-booking.md` entry for the current time window.

#### Scenario: Idle GPU passes claim
- **WHEN** the GPU is idle (utilisation < 10%, memory < 500 MB)
- **THEN** `scripts/claim-gpu.sh` MUST exit 0

#### Scenario: Busy GPU blocks claim
- **WHEN** another user is already training (utilisation > 50%)
- **THEN** the script MUST exit non-zero
- **AND** print the offending PID and user

