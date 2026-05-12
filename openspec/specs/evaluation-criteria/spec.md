# evaluation-criteria Specification

## Purpose
TBD - created by archiving change establish-project-lifecycle-spec. Update Purpose after archive.
## Requirements
### Requirement: Complete Baseline Set
Final evaluation SHALL include exactly 6 conditions: `B0-env-native` (vanilla LunarLander reward), `B1-handcrafted` (a third-party reasonable hand-shaped reward, frozen before evaluation), `B2-gemma-oneshot` (single Gemma-generated reward, EUREKA-style), `B3-hermes-full` (full Hermes-DQN closed loop), `B3-no-memory` (ablation removing the 4-tier memory), and `B3-no-AST` (ablation removing AST-Buffer manager).

#### Scenario: Missing baseline rejection
- **WHEN** the evaluation script detects fewer than 5 completed seeds for any of the 6 conditions
- **THEN** the script MUST exit non-zero
- **AND** MUST print which condition is incomplete and how many seeds are missing

#### Scenario: All 6 conditions complete
- **WHEN** every condition has 5 valid seeds passing the inclusion gate (Requirement: Run Inclusion Gate)
- **THEN** the script MUST proceed to statistical analysis

### Requirement: Statistical Test
Pairwise condition comparisons SHALL use the Mann-Whitney U test (two-tailed, α=0.05) for each primary/secondary metric. Confidence intervals SHALL be computed by bootstrap with 5000 resamples at 95% level. Parametric t-tests SHALL NOT be used because n=5 cannot support a normality assumption.

#### Scenario: Mann-Whitney U on converge_episode
- **WHEN** comparing B3-hermes-full vs B0-env-native on the `converge_episode` metric
- **THEN** the script MUST run `scipy.stats.mannwhitneyu(..., alternative='two-sided')`
- **AND** report the U statistic and the p-value to 4 decimal places

### Requirement: Win Criterion
A condition SHALL be declared "win" over another only when ALL three conditions hold simultaneously: (a) p-value < 0.05, (b) the mean difference is ≥ 10% of the loser's mean, and (c) the bootstrap 95% confidence intervals do not overlap. Failing any single condition demotes the result to "inconclusive".

#### Scenario: Statistical significance without effect size
- **WHEN** p = 0.03 but the mean difference is 6% and CIs overlap
- **THEN** the comparison MUST be reported as "inconclusive", NOT "win"

#### Scenario: Effect size without statistical significance
- **WHEN** the mean difference is 15% but p = 0.08
- **THEN** the comparison MUST be reported as "inconclusive"

#### Scenario: All three conditions met
- **WHEN** p = 0.02, mean difference is 14%, and CIs do not overlap
- **THEN** the better condition MUST be reported as "win"

### Requirement: Primary Metric Hierarchy
`converge_episode` SHALL be the primary metric (sample efficiency). `success_rate` is the secondary metric. `mean_reward_last100` is the tertiary metric. Runs that never converge within 1500 episodes MUST be recorded as `converge_episode = >1500` and SHALL be treated as 1500 for sorting purposes.

#### Scenario: Never-converged run sorting
- **WHEN** a seed never reaches `mean_reward_last100 ≥ 200`
- **THEN** `converge_episode` MUST be recorded as the string `">1500"`
- **AND** for sort comparisons MUST be substituted by the integer 1500

#### Scenario: Tie-break by secondary metric
- **WHEN** two conditions have identical `converge_episode` means
- **THEN** ranking MUST fall back to `success_rate`, then to `mean_reward_last100`

### Requirement: Reporting Table Columns
The final reporting table SHALL contain exactly these columns in this order: `condition`, `n`, `converge_ep mean±CI`, `reward_last100 mean±CI`, `success_rate mean±CI`, `wall_time_min`, `reward_fn_lines`.

#### Scenario: Column order enforced
- **WHEN** the reporting script generates `reports/<exp>/table.md`
- **THEN** the header row MUST match the 7 column names in the exact order above

#### Scenario: CI rendering format
- **WHEN** rendering a metric cell
- **THEN** the format MUST be `mean [low, high]` with values rounded to 1 decimal place (e.g. `1142.3 [1098.1, 1187.7]`)

### Requirement: Training Curve Visualisation
For every condition the system SHALL produce a reward-vs-episode line chart with a shaded 95% bootstrap confidence band. Hermes outer-loop experiments additionally SHALL produce a fitness-vs-iteration chart showing all 5 LLM iterations.

#### Scenario: Output formats and resolution
- **WHEN** the evaluation script finishes
- **THEN** `reports/<exp>/figures/training_curves.png` (300 dpi) MUST exist
- **AND** `reports/<exp>/figures/training_curves.pdf` MUST exist for paper inclusion

#### Scenario: Hermes iteration chart
- **WHEN** the experiment is `B3-hermes-full`
- **THEN** `reports/<exp>/figures/hermes_iterations.png` MUST also exist

### Requirement: Run Inclusion Gate
A run SHALL enter final statistics only when ALL of the following hold: (1) `total_episodes == 1500`, (2) no Python uncaught exception during training, (3) the reward function compiled without error, (4) `mean_reward_last100 > -200`. Runs failing any condition MUST be reported in a separate "excluded" footnote but MUST NOT contaminate the main table.

#### Scenario: Crashed run excluded
- **WHEN** a seed's training raised an uncaught exception
- **THEN** the run MUST be excluded from the main table
- **AND** MUST appear in the `excluded` footnote with reason `crash`

#### Scenario: Divergent but completed run included
- **WHEN** a seed completed 1500 episodes with `mean_reward_last100 = -150`
- **THEN** the run MUST be included in statistics
- **AND** the condition's row footnote MUST note `n=5 (1 divergent)`

### Requirement: Outlier Reporting Without Removal
Statistical outliers SHALL be reported in footnotes (e.g. `n=5 (1 divergent)`) but MUST NOT be removed from the dataset. Cherry-picking is forbidden because it invites reproducibility challenges from reviewers.

#### Scenario: Outlier disclosed
- **WHEN** one of 5 seeds has `mean_reward_last100` more than 2 standard deviations below the others
- **THEN** the row footnote MUST disclose `1 outlier (z=-2.4) retained`

### Requirement: Compute Cost Disclosure
Every condition row MUST include `wall_time_min` (median across the 5 seeds). The end of the report MUST state total GPU-hours consumed and the hardware declaration `Hardware: NVIDIA RTX 4090 × 1, Windows 11, Python 3.11, CUDA 12.1`.

#### Scenario: Total compute reported
- **WHEN** the report is finalised
- **THEN** the report's footer MUST contain a line of the form `Total compute: NN.N GPU-hours on NVIDIA RTX 4090`

### Requirement: Reproducibility Seven-Pack
Every result row in the report MUST be accompanied by 7 reproducibility artifacts, available either inline or via path reference: (1) git commit SHA, (2) the 5 seed integers, (3) `reward_fn.py`, (4) `config.json`, (5) `episodes.jsonl`, (6) `pyproject.toml` snapshot, (7) Gemma prompt+response log (PII-scrubbed) for any LLM-generated reward.

#### Scenario: Reproducibility seven-pack complete
- **WHEN** the final report is finalised for the paper
- **THEN** each row MUST cite paths or content for all 7 artifacts
- **AND** any missing artifact MUST block report publication

#### Scenario: LLM log PII-scrubbed
- **WHEN** a Gemma prompt/response log is committed to the repo
- **THEN** it MUST have been processed by the scrubbing pipeline that removes email addresses, API keys, and absolute file paths

