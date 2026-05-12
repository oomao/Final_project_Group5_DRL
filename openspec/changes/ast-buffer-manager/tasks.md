## 1. Buffer package skeleton

- [x] 1.1 Create `hermes_dqn/buffer/` with `__init__.py` re-exporting `RewardDiff`, `BufferAction`, `diff_rewards`, `decide_policy`, `apply_policy`
- [x] 1.2 Add module docstrings explaining the L1/L2/L3-style threat-irrelevant scope of this module (this is about *learning hygiene*, not security)

## 2. AST diff classifier

- [x] 2.1 Implement `hermes_dqn/buffer/ast_diff.py::RewardDiff` frozen dataclass with fields `kind: str`, `similarity: float`, `diff_summary: str`
- [x] 2.2 Implement `_ast_signature(tree)` returning a list of canonical node tokens:
  - `type(node).__name__` for every node
  - `ast.Name`: append `.id`
  - `ast.Attribute`: append `.attr`
  - `ast.Constant`: append `"<NUM>"` for int/float, `repr(value)` for str/bool/None
- [x] 2.3 Implement `diff_rewards(old_src, new_src) -> RewardDiff`:
  - 2.3.1 Bytes-equal short-circuit → `IDENTICAL`
  - 2.3.2 Try `ast.parse` both; on SyntaxError of either → `TOTAL_REWRITE` (don't raise)
  - 2.3.3 If signatures equal → `NUMERIC_DIFF` (similarity=1.0)
  - 2.3.4 Otherwise compute `difflib.SequenceMatcher(None, sig_old, sig_new).ratio()`
  - 2.3.5 ratio > 0.7 → `STRUCTURAL_DIFF`; else `TOTAL_REWRITE`
- [x] 2.4 Populate `diff_summary`: when NUMERIC_DIFF list changed constants; when STRUCTURAL or TOTAL include `len(sig_old)` / `len(sig_new)` / ratio

## 3. Buffer policy

- [x] 3.1 Implement `hermes_dqn/buffer/policy.py::BufferAction` Enum with members `KEEP`, `DECAY`, `CLEAR`
- [x] 3.2 Implement `decide_policy(diff: RewardDiff) -> BufferAction` per the lookup table in design.md
- [x] 3.3 Implement `apply_policy(buffer, action, decay_factor=0.5)` dispatching to `buffer.decay_weights` or `buffer.clear()`; raise `ValueError` on unknown action

## 4. Extend ReplayBuffer

- [x] 4.1 Add `self._weights = np.ones(capacity, dtype=np.float32)` to `__init__`
- [x] 4.2 In `push`: set `self._weights[i] = 1.0` for the overwritten slot
- [x] 4.3 Modify `sample(batch_size)`:
  - 4.3.1 Add fast-path: if `np.all(self._weights[: self._size] == 1.0)`, use legacy `self._rng.integers(0, self._size, size=batch_size)` for byte-deterministic backward-compat
  - 4.3.2 Slow path: compute `probs = weights/weights.sum()` and use `self._rng.choice(self._size, size=batch_size, p=probs, replace=True)`
- [x] 4.4 Implement `decay_weights(factor: float) -> None`: `self._weights[: self._size] *= factor`
- [x] 4.5 Implement `clear() -> None`: zero all 6 arrays (incl. `_weights`), `_idx = _size = 0`. Do NOT reset `_rng`.
- [x] 4.6 Implement `save(path) -> None`:
  - 4.6.1 Use `np.savez_compressed`
  - 4.6.2 Save only `[:_size]` slices of each array (size efficiency)
  - 4.6.3 Save `_idx`, `_size`, `capacity` as scalars
  - 4.6.4 Save RNG state via `np.array([self._rng.bit_generator.state], dtype=object)`
- [x] 4.7 Implement `load(path) -> None`:
  - 4.7.1 Load via `np.load(path, allow_pickle=True)`
  - 4.7.2 Reconstruct full-capacity arrays, fill `[:size]` from saved slices, leave rest as 0
  - 4.7.3 Restore `_idx`, `_size`
  - 4.7.4 Restore RNG state via `self._rng.bit_generator.state = saved_state.item()`

## 5. Unit tests

- [x] 5.1 Create `tools/_smoke_ast_buffer.py` running these cases:
  - 5.1.1 `diff_rewards` IDENTICAL: same string twice → `IDENTICAL`, similarity=1.0
  - 5.1.2 `diff_rewards` NUMERIC: same body with `0.1` → `0.2` → `NUMERIC_DIFF`, similarity=1.0
  - 5.1.3 `diff_rewards` STRUCTURAL: add one extra `+ leg_bonus` term → `STRUCTURAL_DIFF`, similarity in (0.7, 1.0)
  - 5.1.4 `diff_rewards` TOTAL: replace body with `return abs(obs[0])` → `TOTAL_REWRITE`, similarity <= 0.7
  - 5.1.5 `diff_rewards` unparseable: malformed `def reward` → returns TOTAL_REWRITE without raising
  - 5.1.6 `decide_policy` all four kinds → correct BufferAction
  - 5.1.7 `apply_policy(KEEP)` → no state change
  - 5.1.8 `apply_policy(DECAY, 0.5)` on full-weight buffer → all existing weights == 0.5
  - 5.1.9 `apply_policy(CLEAR)` → `len(buffer) == 0`
  - 5.1.10 Backward-compat: build a fresh ReplayBuffer, push 100 transitions, sample 10 with seed=0 → returns same indices as legacy ReplayBuffer would (use a saved fixture)
  - 5.1.11 Decay sampling probability: push 1000 transitions at weight 1.0, decay_weights(0.5), push 1000 more at weight 1.0 → sample 10000 times → ~ 2/3 of samples should come from indices >= 1000

## 6. Persistence smoke

- [x] 6.1 `tools/_smoke_buffer_persist.py`:
  - 6.1.1 Build buffer, push 5000 random transitions, `sample(64)` to advance RNG, save to `/tmp/buf.npz`
  - 6.1.2 New ReplayBuffer with same capacity, `load(/tmp/buf.npz)`
  - 6.1.3 Assert `_obs`, `_actions`, `_rewards`, `_next_obs`, `_dones`, `_weights` arrays equal `[:5000]`
  - 6.1.4 Assert `_idx`, `_size` equal
  - 6.1.5 Both buffers `sample(64)` → indices arrays MUST be byte-identical (proves RNG state restored)

## 7. Backward-compat smoke (no regression to baseline)

- [x] 7.1 Run `python -m hermes_dqn.training.train --reward-source env --episodes 10 --seed 42 --out-dir runs/smoke_ast_v3_env`
- [x] 7.2 Compare first 10 episode `return` values to `runs/baseline_seed42/episodes.jsonl` first 10 rows; MUST be byte-identical (the new `_weights` infrastructure must not perturb deterministic training)
- [x] 7.3 Compare `reward_fn_sha256` and `env_native_mean` ranges to confirm no semantic drift

## 8. Wrap-up

- [x] 8.1 `openspec validate ast-buffer-manager --strict` passes
- [x] 8.2 All scenarios across `specs/ast-buffer-manager/spec.md` verified by tests in 5.x / 6.x / 7.x
- [x] 8.3 No regression to `bootstrap-dqn-baseline`'s deterministic guarantee (7.x evidence)
- [x] 8.4 Ready to `/opsx:archive` once next handover is written
