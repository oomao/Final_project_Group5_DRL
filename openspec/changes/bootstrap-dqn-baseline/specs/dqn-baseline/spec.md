## ADDED Requirements

### Requirement: Train DQN on LunarLander-v3 end-to-end
The system SHALL provide a single command that trains a vanilla DQN agent on Gymnasium `LunarLander-v3` from random initialization to convergence and writes all artifacts to a timestamped run directory.

#### Scenario: Default training run
- **WHEN** the user runs `python -m hermes_dqn.training.train`
- **THEN** the system creates `runs/<YYYY-MM-DD_HH-MM-SS>/` containing `config.json`, `episodes.jsonl`, and `model_final.pt`
- **AND** training proceeds for the configured number of episodes (default 1500) without raising

#### Scenario: Episode count override
- **WHEN** the user runs `python -m hermes_dqn.training.train --episodes 50`
- **THEN** training stops after 50 episodes regardless of convergence
- **AND** the `config.json` records `episodes: 50`

### Requirement: Vanilla DQN architecture
The system SHALL implement DQN with all four standard components: Q-network, target network, uniform replay buffer, and ε-greedy exploration. Rainbow extensions (Double / Dueling / Prioritized Replay) SHALL NOT be included in this baseline.

#### Scenario: Target network sync
- **WHEN** the env step counter is an exact multiple of the target-update interval
- **THEN** the target network's weights are hard-copied from the online Q-network

#### Scenario: Epsilon decay
- **WHEN** training is in progress
- **THEN** ε decays linearly from 1.0 to 0.01 over the configured decay window
- **AND** after the decay window, ε remains at 0.01 for the rest of training

#### Scenario: Replay-buffer warm-up
- **WHEN** the buffer holds fewer transitions than the configured `train_start` threshold
- **THEN** the agent collects experience but performs no gradient updates

### Requirement: Reproducible training
The system SHALL produce identical episode-return sequences across runs that share the same seed, code version, and OS.

#### Scenario: Same seed twice
- **WHEN** the user runs training twice with `--seed 42` on the same machine and code version
- **THEN** both `episodes.jsonl` files contain identical `return` values for every episode

#### Scenario: Seed is persisted
- **WHEN** any training run completes
- **THEN** `config.json` contains the seed used, the env name, and every hyperparameter

### Requirement: CPU and GPU support
The system SHALL train on CPU by default and automatically use CUDA if `torch.cuda.is_available()` is True, without requiring code changes.

#### Scenario: CPU-only machine
- **WHEN** training is launched on a machine without CUDA
- **THEN** all tensors and the Q-network live on `cpu` and training completes successfully

#### Scenario: CUDA available
- **WHEN** training is launched on a machine where `torch.cuda.is_available()` returns True
- **THEN** the Q-network and target network are moved to `cuda:0` and training uses GPU
