# fitness-evaluation Specification

## Purpose
TBD - created by archiving change bootstrap-dqn-baseline. Update Purpose after archive.
## Requirements
### Requirement: FitnessReport data shape
The system SHALL define a `FitnessReport` dataclass with exactly these fields: `converge_episode: int | None`, `mean_reward_last100: float`, `success_rate: float`, `total_episodes: int`, `seed: int`.

#### Scenario: Field availability
- **WHEN** a `FitnessReport` instance is created
- **THEN** all five fields are accessible as attributes
- **AND** `converge_episode` is either an integer episode index or `None`

### Requirement: FitnessEvaluator reads JSONL logs
The system SHALL provide a `FitnessEvaluator.evaluate(jsonl_path)` method that consumes an `episodes.jsonl` file produced by training and returns a `FitnessReport`.

#### Scenario: Standard converged run
- **WHEN** `evaluate()` is called on an `episodes.jsonl` whose last 100 episodes have mean return ≥ 200.0
- **THEN** the returned report has `mean_reward_last100 >= 200.0`
- **AND** `converge_episode` equals the first episode index where the rolling 100-episode mean crossed the threshold
- **AND** `success_rate` equals the fraction of the last 100 episodes whose return is ≥ 200.0

#### Scenario: Run that never converges
- **WHEN** `evaluate()` is called on a log whose rolling 100-episode mean never reaches 200.0
- **THEN** the returned report has `converge_episode is None`
- **AND** `mean_reward_last100` is still computed from the final 100 episodes
- **AND** `success_rate` is still computed from the final 100 episodes

#### Scenario: Short run (fewer than 100 episodes)
- **WHEN** `evaluate()` is called on a log containing fewer than 100 episode rows
- **THEN** `mean_reward_last100` is computed over all available episodes
- **AND** `total_episodes` reflects the actual count
- **AND** `success_rate` is computed over the same available episodes

### Requirement: Configurable success threshold and window
`FitnessEvaluator` SHALL accept `success_threshold: float = 200.0` and `window: int = 100` at construction; later changes that target environments with different success criteria can override these without modifying the evaluator.

#### Scenario: Override threshold
- **WHEN** `FitnessEvaluator(success_threshold=150.0, window=50)` is constructed
- **THEN** `evaluate()` uses 150.0 as the success threshold and 50 as the rolling window
- **AND** `success_rate` is the fraction of the last 50 episodes with return ≥ 150.0

