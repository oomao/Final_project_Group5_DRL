## ADDED Requirements

### Requirement: Pairwise Mann-Whitney comparison
`tools/compare_conditions.py` SHALL accept `--exp <name>` and `--conditions A,B,C,...` and compute, for each pair of conditions, a two-sided Mann-Whitney U test (α=0.05) on the per-seed `env_native_mean` values from the LAST iteration of each `(condition, seed)` run.

#### Scenario: Standard pairwise computation
- **WHEN** `python tools/compare_conditions.py --exp final --conditions B0,B1,B2,B3` is run on a populated `runs/final/` tree
- **THEN** the script MUST emit a pairwise p-value matrix (4 × 4) where each off-diagonal entry is the Mann-Whitney U p-value
- **AND** each diagonal entry MUST be `1.000` (or marked `—`)

#### Scenario: Missing seeds are reported, not silently skipped
- **WHEN** condition `B3` has only 4 of the expected 5 seeds present
- **THEN** the script MUST print a warning naming the missing seed(s)
- **AND** the comparison MUST still run with n=4 for that condition
- **AND** the report MUST footnote the n shortage

### Requirement: Bootstrap confidence intervals
For each condition, the script SHALL compute a 95% bootstrap confidence interval over `env_native_mean` via 5000 resamples (with replacement) of the per-seed values. For each pair, the script SHALL also compute a bootstrap CI for the mean difference.

#### Scenario: CI is reported per condition
- **WHEN** the script processes condition `B3-hermes-full` with 5 seeds whose env_native_mean values are [205, 210, 218, 225, 220]
- **THEN** the report MUST include `B3-hermes-full: mean=215.6, 95% CI=[X.X, Y.Y]` where the CI is computed via 5000 bootstrap resamples
- **AND** the random seed for bootstrap MUST be fixed (e.g., 12345) so the report is reproducible

### Requirement: Win determination requires three conditions
The script SHALL mark condition A as "winning over" condition B only when ALL THREE conditions hold simultaneously: (1) Mann-Whitney U two-sided p < 0.05; (2) `(mean(A) - mean(B)) / mean(B) >= 0.10` (10% relative gap); (3) the 95% bootstrap CIs of A and B do not overlap. If any condition fails, the pair MUST be marked `inconclusive`.

#### Scenario: All three conditions met -> win
- **WHEN** A's mean=210, B's mean=160, p=0.01, A's CI=[200, 220], B's CI=[150, 170]
- **THEN** the report MUST mark `A wins over B`

#### Scenario: p significant but CI overlaps -> inconclusive
- **WHEN** A's mean=215, B's mean=200, p=0.03, A's CI=[195, 235], B's CI=[180, 220]
- **THEN** the report MUST mark this pair as `inconclusive` and explicitly list `reason: CIs overlap`

#### Scenario: Gap below 10% -> inconclusive
- **WHEN** A's mean=210, B's mean=200, p=0.001, CIs non-overlapping
- **THEN** the report MUST mark this pair as `inconclusive` and list `reason: effect size 5% < 10% threshold`

### Requirement: Reporting outputs
The script SHALL emit `reports/<exp>/comparison_report.md` with: (a) a summary table per condition (`n`, `mean ± 95% CI`, success_rate ± CI, crash_rate, mean_wall_time, n_divergent footnote); (b) a pairwise p-value + win/loss matrix; (c) a `## Outliers` section listing divergent runs per condition with their env_native_mean. AND the script SHALL emit `reports/<exp>/figures/training_curves.png` and `reports/<exp>/figures/iteration_fitness.png` at 300 dpi.

#### Scenario: Summary table format
- **WHEN** the report is produced
- **THEN** the summary table MUST have one row per condition
- **AND** columns MUST include condition name, n, mean env_native_mean with bootstrap CI, success rate, crash rate, mean wall-time (minutes), and a `divergent` count
- **AND** rows MUST be sorted by mean env_native_mean descending

#### Scenario: Outlier disclosure required
- **WHEN** any condition has at least 1 divergent run (final env_native_mean < -200 or status="failed")
- **THEN** the report's `## Outliers` section MUST list the run's path, condition, seed, and env_native_mean
- **AND** that run MUST NOT be silently excluded from the summary statistics (per evaluation-criteria R7)

#### Scenario: Figure files exist and embed proper labels
- **WHEN** the script completes
- **THEN** `training_curves.png` MUST exist with axes labeled "Episode" and "Return", a legend mapping color to condition, and a title including the exp_name
- **AND** `iteration_fitness.png` MUST plot env_native_mean as a function of LLM iteration (1..N), one line per `(condition, seed)`, with axes "LLM iteration" and "env_native_mean"

### Requirement: Compute cost is reported
The final report SHALL include a `## Compute` section listing total wall-clock GPU hours, breakdown per condition, and a hardware declaration (`NVIDIA RTX 4090 × 1`).

#### Scenario: Compute section present
- **WHEN** the report is generated
- **THEN** it MUST end with a `## Compute` section
- **AND** that section MUST list each condition's total wall-time aggregated from summary.json entries
- **AND** MUST include the line `Hardware: NVIDIA RTX 4090 × 1` (matching env-setup spec)
