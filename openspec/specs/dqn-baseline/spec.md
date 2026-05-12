# dqn-baseline Specification

## Purpose
TBD - created by archiving change bootstrap-dqn-baseline. Update Purpose after archive.
## Requirements
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
The system SHALL implement DQN with all four standard components: Q-network, target network, uniform replay buffer, and ε-greedy exploration. Rainbow extensions (Double / Dueling / Prioritized Replay) SHALL NOT be included in this baseline. The replay buffer SHALL additionally support optional per-sample sampling weights (introduced by `ast-buffer-manager`); when all weights equal 1.0 the buffer's `sample()` SHALL behave byte-identically to the original uniform implementation.

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

#### Scenario: Uniform-weights fast path preserves determinism
- **WHEN** the buffer's `_weights[: _size]` are all exactly 1.0 (the default state, e.g. for `bootstrap-dqn-baseline` and earlier runs)
- **THEN** `sample()` MUST take the legacy fast path using `self._rng.integers(0, _size, size=batch_size)`
- **AND** the produced index sequence MUST be byte-identical to what `bootstrap-dqn-baseline`-era runs produced for the same seed

#### Scenario: Decayed weights take the weighted path
- **WHEN** `decay_weights(0.5)` has been called and `_weights` no longer all equal 1.0
- **THEN** `sample()` MUST take the weighted path using `np.random.Generator.choice(_size, p=weights/weights.sum())`
- **AND** the per-sample selection probability MUST be proportional to its weight

### Requirement: Reproducible training
The system SHALL produce identical episode-return sequences across runs that share the same seed, code version, and OS. The replay buffer's persistence functions (`save`, `load`) SHALL preserve all RNG state needed to reproduce subsequent `sample()` sequences byte-identically across processes.

#### Scenario: Same seed twice
- **WHEN** the user runs training twice with `--seed 42` on the same machine and code version
- **THEN** both `episodes.jsonl` files contain identical `return` values for every episode

#### Scenario: Seed is persisted
- **WHEN** any training run completes
- **THEN** `config.json` contains the seed used, the env name, and every hyperparameter

#### Scenario: Buffer save/load preserves RNG state
- **WHEN** a buffer with 5000 transitions is saved to disk via `buffer.save(path)`, then loaded into a fresh `ReplayBuffer` instance via `buffer2.load(path)`
- **THEN** `buffer2.sample(64)` MUST produce a Batch byte-identical to what the original `buffer.sample(64)` would have produced at that point
- **AND** all 5 arrays + `_weights` + `_idx` + `_size` + RNG state MUST match exactly

### Requirement: CPU and GPU support
The system SHALL train on CPU by default and automatically use CUDA if `torch.cuda.is_available()` is True, without requiring code changes.

#### Scenario: CPU-only machine
- **WHEN** training is launched on a machine without CUDA
- **THEN** all tensors and the Q-network live on `cpu` and training completes successfully

#### Scenario: CUDA available
- **WHEN** training is launched on a machine where `torch.cuda.is_available()` returns True
- **THEN** the Q-network and target network are moved to `cuda:0` and training uses GPU

### Requirement: ReplayBuffer persistence and reset
The replay buffer SHALL expose `save(path)`, `load(path)`, `decay_weights(factor)`, and `clear()` methods. `save` and `load` SHALL round-trip the full buffer state including weights and RNG. `decay_weights(factor)` SHALL multiply all existing samples' weights by `factor` without touching newly pushed samples (which always start at weight 1.0). `clear()` SHALL reset `_idx` and `_size` to 0 and zero the underlying arrays.

#### Scenario: Save then load yields identical buffer
- **WHEN** `buffer1.save("/tmp/b.npz")` is called after pushing 5000 transitions and `buffer2 = ReplayBuffer(...); buffer2.load("/tmp/b.npz")` is then called
- **THEN** `buffer2._obs[:5000]`, `_actions[:5000]`, `_rewards[:5000]`, `_next_obs[:5000]`, `_dones[:5000]`, `_weights[:5000]`, `_idx`, `_size`, and RNG state MUST equal `buffer1`'s

#### Scenario: decay_weights only affects existing samples
- **WHEN** a buffer has 1000 transitions at weight 1.0, `decay_weights(0.5)` is called, then 500 new transitions are pushed
- **THEN** `_weights[:1000]` MUST equal `0.5`
- **AND** `_weights[1000:1500]` MUST equal `1.0`

#### Scenario: clear resets buffer to empty
- **WHEN** `buffer.clear()` is called on a buffer with N transitions
- **THEN** `len(buffer)` MUST be 0
- **AND** `_idx` MUST be 0
- **AND** subsequent `push` calls MUST start writing at index 0

