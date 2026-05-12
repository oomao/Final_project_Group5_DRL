## ADDED Requirements

### Requirement: RewardDiff classification
The system SHALL provide `diff_rewards(old_src: str, new_src: str) -> RewardDiff` in `hermes_dqn/buffer/ast_diff.py` that classifies two reward-function source strings into exactly one of four kinds: `"IDENTICAL"`, `"NUMERIC_DIFF"`, `"STRUCTURAL_DIFF"`, `"TOTAL_REWRITE"`. The returned `RewardDiff` SHALL be a frozen dataclass with fields `kind: str`, `similarity: float` (in [0.0, 1.0]), and `diff_summary: str`.

#### Scenario: Identical sources
- **WHEN** `diff_rewards(src, src)` is called with byte-identical strings
- **THEN** the result MUST have `kind == "IDENTICAL"`
- **AND** `similarity == 1.0`

#### Scenario: Numeric coefficient change only
- **WHEN** `old_src` defines `reward(...) -> 0.1 * abs(x)` and `new_src` defines `reward(...) -> 0.2 * abs(x)`
- **THEN** the result MUST have `kind == "NUMERIC_DIFF"`
- **AND** `similarity == 1.0` (AST structures are equal once numeric constants are placeholder-normalized)

#### Scenario: Structural change with high similarity
- **WHEN** `new_src` adds one extra term (e.g. an extra leg-contact bonus) to a reward that otherwise has the same shape
- **THEN** the result MUST have `kind == "STRUCTURAL_DIFF"`
- **AND** `similarity` MUST be in (0.7, 1.0)

#### Scenario: Total rewrite
- **WHEN** `new_src` replaces the entire reward body with a wildly different expression (different variables, different control flow)
- **THEN** the result MUST have `kind == "TOTAL_REWRITE"`
- **AND** `similarity` MUST be <= 0.7

#### Scenario: Unparseable input falls back to TOTAL_REWRITE
- **WHEN** `old_src` is syntactically broken (e.g. missing colon) but `new_src` is valid
- **THEN** the result MUST have `kind == "TOTAL_REWRITE"` (conservative fallback)
- **AND** the function MUST NOT raise `SyntaxError`

### Requirement: BufferAction enum and decide_policy
The system SHALL define `BufferAction = Enum("BufferAction", "KEEP DECAY CLEAR")` in `hermes_dqn/buffer/policy.py` and provide `decide_policy(diff: RewardDiff) -> BufferAction` mapping diff kinds to actions per this table: `IDENTICAL/NUMERIC_DIFF → KEEP`, `STRUCTURAL_DIFF → DECAY`, `TOTAL_REWRITE → CLEAR`.

#### Scenario: All four kinds map to actions
- **WHEN** `decide_policy` is called on each of the four RewardDiff kinds
- **THEN** the function MUST return `KEEP` for IDENTICAL, `KEEP` for NUMERIC_DIFF, `DECAY` for STRUCTURAL_DIFF, `CLEAR` for TOTAL_REWRITE

#### Scenario: BufferAction is import-safe from the package root
- **WHEN** a developer imports `from hermes_dqn.buffer import BufferAction`
- **THEN** the import MUST succeed and `BufferAction.KEEP`, `BufferAction.DECAY`, `BufferAction.CLEAR` MUST all be accessible

### Requirement: apply_policy mutates buffer in place
The system SHALL provide `apply_policy(buffer, action, decay_factor: float = 0.5) -> None` in `hermes_dqn/buffer/rebuild.py` that mutates `buffer` in place according to `action`. `KEEP` is a no-op. `DECAY` calls `buffer.decay_weights(decay_factor)`. `CLEAR` calls `buffer.clear()`.

#### Scenario: KEEP is a no-op
- **WHEN** `apply_policy(buffer, BufferAction.KEEP)` is called on a buffer with 1000 transitions
- **THEN** the buffer's `__len__` MUST remain 1000
- **AND** no internal state (weights, idx, size, RNG) MUST have changed

#### Scenario: DECAY scales existing weights
- **WHEN** `apply_policy(buffer, BufferAction.DECAY, decay_factor=0.5)` is called on a buffer with 1000 transitions whose weights are all 1.0
- **THEN** all 1000 existing weights MUST equal 0.5 afterwards
- **AND** `len(buffer)` MUST still be 1000

#### Scenario: CLEAR empties the buffer
- **WHEN** `apply_policy(buffer, BufferAction.CLEAR)` is called
- **THEN** `len(buffer)` MUST be 0
- **AND** subsequent `buffer.sample(64)` MUST raise OR (if `_size==0` permitted) return an empty Batch — the implementation chooses one and documents it

#### Scenario: Unknown action raises
- **WHEN** `apply_policy(buffer, some_unknown_value)` is called
- **THEN** a `ValueError` MUST be raised naming the offending value

## MODIFIED Requirements

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
