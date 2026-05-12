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
