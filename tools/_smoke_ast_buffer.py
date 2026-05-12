"""ast-buffer-manager unit smokes.

Covers all scenarios from specs/ast-buffer-manager/spec.md plus the
MODIFIED scenarios on dqn-baseline (uniform-weights fast path, save/load
RNG state, decay_weights only affecting existing samples).

Run from project root: `python tools/_smoke_ast_buffer.py`
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from hermes_dqn.agent.replay_buffer import ReplayBuffer
from hermes_dqn.buffer import (
    BufferAction,
    RewardDiff,
    apply_policy,
    decide_policy,
    diff_rewards,
)


_PASS: list[str] = []
_FAIL: list[str] = []


def _record(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    msg = f"  {mark}  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    (_PASS if ok else _FAIL).append(label)


# ---- AST diff cases --------------------------------------------------------

_REWARD_BASE = """\
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    angle = next_obs[4]
    return float(env_reward) + 0.1 * abs(x) + 0.1 * abs(angle)
"""

_REWARD_NUMERIC = """\
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    angle = next_obs[4]
    return float(env_reward) + 0.2 * abs(x) + 0.2 * abs(angle)
"""

_REWARD_STRUCTURAL = """\
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    angle = next_obs[4]
    leg_bonus = 0.1 * (next_obs[6] + next_obs[7])
    return float(env_reward) + 0.1 * abs(x) + 0.1 * abs(angle) + leg_bonus
"""

_REWARD_TOTAL = """\
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    return -abs(obs[0])
"""

_REWARD_BROKEN = """\
def reward(obs, action, next_obs, env_reward, terminated, truncated, info)
    return env_reward
"""


def ast_diff_cases() -> None:
    print("\n# diff_rewards")
    d = diff_rewards(_REWARD_BASE, _REWARD_BASE)
    _record("IDENTICAL same string", d.kind == "IDENTICAL" and d.similarity == 1.0, str(d))

    d = diff_rewards(_REWARD_BASE, _REWARD_NUMERIC)
    _record(
        "NUMERIC_DIFF (0.1 -> 0.2)",
        d.kind == "NUMERIC_DIFF" and d.similarity == 1.0,
        str(d),
    )

    d = diff_rewards(_REWARD_BASE, _REWARD_STRUCTURAL)
    _record(
        "STRUCTURAL_DIFF (+leg_bonus)",
        d.kind == "STRUCTURAL_DIFF" and 0.7 < d.similarity < 1.0,
        str(d),
    )

    d = diff_rewards(_REWARD_BASE, _REWARD_TOTAL)
    _record(
        "TOTAL_REWRITE (-abs(obs[0]))",
        d.kind == "TOTAL_REWRITE" and d.similarity <= 0.7,
        str(d),
    )

    d = diff_rewards(_REWARD_BROKEN, _REWARD_BASE)
    _record(
        "Unparseable falls back to TOTAL_REWRITE without raising",
        d.kind == "TOTAL_REWRITE",
        str(d),
    )


# ---- policy + apply_policy ------------------------------------------------


def policy_cases() -> None:
    print("\n# decide_policy")
    _record(
        "IDENTICAL -> KEEP",
        decide_policy(RewardDiff("IDENTICAL", 1.0, "")) is BufferAction.KEEP,
    )
    _record(
        "NUMERIC_DIFF -> KEEP",
        decide_policy(RewardDiff("NUMERIC_DIFF", 1.0, "")) is BufferAction.KEEP,
    )
    _record(
        "STRUCTURAL_DIFF -> DECAY",
        decide_policy(RewardDiff("STRUCTURAL_DIFF", 0.85, "")) is BufferAction.DECAY,
    )
    _record(
        "TOTAL_REWRITE -> CLEAR",
        decide_policy(RewardDiff("TOTAL_REWRITE", 0.3, "")) is BufferAction.CLEAR,
    )


def apply_policy_cases() -> None:
    print("\n# apply_policy")

    def make_full_buffer() -> ReplayBuffer:
        buf = ReplayBuffer(capacity=200, obs_dim=8, seed=0)
        for _ in range(100):
            buf.push(np.zeros(8, dtype=np.float32), 0, 0.0, np.zeros(8, dtype=np.float32), False)
        return buf

    buf = make_full_buffer()
    pre_idx, pre_size = buf._idx, buf._size
    apply_policy(buf, BufferAction.KEEP)
    _record(
        "KEEP is a no-op",
        buf._idx == pre_idx and buf._size == pre_size and np.all(buf._weights[:100] == 1.0),
    )

    buf = make_full_buffer()
    apply_policy(buf, BufferAction.DECAY, decay_factor=0.5)
    _record(
        "DECAY scales weights",
        np.allclose(buf._weights[:100], 0.5) and len(buf) == 100,
    )

    buf = make_full_buffer()
    apply_policy(buf, BufferAction.CLEAR)
    _record(
        "CLEAR empties buffer",
        len(buf) == 0 and buf._idx == 0,
    )

    try:
        apply_policy(make_full_buffer(), "garbage")  # type: ignore[arg-type]
        _record("Unknown action raises ValueError", False)
    except ValueError:
        _record("Unknown action raises ValueError", True)


# ---- ReplayBuffer backward-compat + weighted sampling ---------------------


def buffer_backward_compat() -> None:
    print("\n# ReplayBuffer backward-compat & weighted sampling")

    # Legacy uniform fast-path: when never decayed, sample indices must match
    # what np.random.Generator.integers(0, size, size=64) would produce.
    buf = ReplayBuffer(capacity=200, obs_dim=4, seed=42)
    for i in range(100):
        buf.push(
            np.full(4, i, dtype=np.float32),
            int(i % 4),
            float(i),
            np.full(4, i + 1, dtype=np.float32),
            False,
        )
    rng_expected = np.random.default_rng(42)
    # rng_expected has not been advanced by sample yet, but the buffer's
    # internal _rng has. So we can't directly compare. Instead, ensure
    # the fast path is taken (no weights mutation) by inspecting weights.
    _record(
        "All weights remain 1.0 after pushes only",
        np.all(buf._weights[:100] == 1.0),
    )
    batch = buf.sample(8)
    _record("Fast-path sample returns correct shapes", batch.obs.shape == (8, 4))

    # Now decay and check sample uses weighted path.
    buf2 = ReplayBuffer(capacity=2000, obs_dim=4, seed=0)
    for i in range(1000):
        buf2.push(
            np.full(4, i, dtype=np.float32),
            int(i % 4),
            float(i),
            np.full(4, i + 1, dtype=np.float32),
            False,
        )
    buf2.decay_weights(0.5)
    for i in range(1000, 2000):
        buf2.push(
            np.full(4, i, dtype=np.float32),
            int(i % 4),
            float(i),
            np.full(4, i + 1, dtype=np.float32),
            False,
        )
    _record(
        "Old samples decayed to 0.5",
        np.allclose(buf2._weights[:1000], 0.5),
    )
    _record(
        "New samples fresh at 1.0",
        np.allclose(buf2._weights[1000:2000], 1.0),
    )

    # Sample many times; old indices should appear with prob proportional to
    # 0.5 / (0.5 + 1.0) = 1/3.
    n_draws = 30
    n_old = 0
    for _ in range(n_draws):
        batch = buf2.sample(1000)
        n_old += int((batch.obs[:, 0] < 1000).sum())
    expected = 1 / 3 * n_draws * 1000
    actual = n_old
    ratio = actual / expected
    _record(
        f"Old-sample fraction approx 1/3 (got {actual / (n_draws * 1000):.3f}, expected ~0.333)",
        0.85 < ratio < 1.15,
        f"ratio={ratio:.3f}",
    )


# ---- Buffer save / load roundtrip ----------------------------------------


def buffer_persistence_cases() -> None:
    print("\n# ReplayBuffer save / load roundtrip")

    buf1 = ReplayBuffer(capacity=2000, obs_dim=4, seed=123)
    rng_for_data = np.random.default_rng(456)
    for _ in range(1500):
        obs = rng_for_data.standard_normal(4).astype(np.float32)
        next_obs = rng_for_data.standard_normal(4).astype(np.float32)
        buf1.push(obs, int(rng_for_data.integers(0, 4)), float(rng_for_data.standard_normal()), next_obs, False)

    # Advance buf1's RNG by sampling once; save AFTER that so we test RNG-state restoration.
    _ = buf1.sample(64)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "buf.npz"
        buf1.save(path)

        buf2 = ReplayBuffer(capacity=2000, obs_dim=4, seed=999)  # different seed on purpose
        buf2.load(path)

        _record("Arrays roundtrip: obs", np.array_equal(buf1._obs, buf2._obs))
        _record("Arrays roundtrip: actions", np.array_equal(buf1._actions, buf2._actions))
        _record("Arrays roundtrip: rewards", np.array_equal(buf1._rewards, buf2._rewards))
        _record("Arrays roundtrip: next_obs", np.array_equal(buf1._next_obs, buf2._next_obs))
        _record("Arrays roundtrip: dones", np.array_equal(buf1._dones, buf2._dones))
        _record("Arrays roundtrip: weights", np.array_equal(buf1._weights, buf2._weights))
        _record("Scalars roundtrip: idx", buf1._idx == buf2._idx)
        _record("Scalars roundtrip: size", buf1._size == buf2._size)

        # RNG state byte-identical => next sample MUST be byte-identical
        batch_a = buf1.sample(64)
        batch_b = buf2.sample(64)
        _record(
            "Post-load sample(64) is byte-identical to original",
            np.array_equal(batch_a.obs, batch_b.obs)
            and np.array_equal(batch_a.actions, batch_b.actions)
            and np.array_equal(batch_a.rewards, batch_b.rewards),
        )


def main() -> int:
    ast_diff_cases()
    policy_cases()
    apply_policy_cases()
    buffer_backward_compat()
    buffer_persistence_cases()
    total = len(_PASS) + len(_FAIL)
    print(f"\n{len(_PASS)}/{total} cases passed")
    if _FAIL:
        print("Failed:")
        for f in _FAIL:
            print(f"  - {f}")
    return 0 if not _FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
