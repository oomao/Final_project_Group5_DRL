"""Sandbox L2 unit smokes — no API needed.

Tests:
1. Valid reward passes
2. Syntax error -> RewardCompileError(stage="syntax-error") via subprocess
3. import statement -> RewardCompileError(stage="ast-import-rejected") via subprocess
4. Infinite loop -> RewardCompileError(stage="subprocess-timeout") within 12s
5. Bad arity -> RewardCompileError(stage="signature-arity")
"""

from __future__ import annotations

import time

from hermes_dqn.llm import RewardCompileError, compile_reward


VALID_SRC = """
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    return float(env_reward) + 0.1 * abs(next_obs[4])
"""

SYNTAX_BAD = """
def reward(obs, action, next_obs, env_reward, terminated, truncated, info)
    return env_reward
"""

IMPORT_BAD = """
import os
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    return float(env_reward)
"""

INFINITE_LOOP = """
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    while True:
        pass
    return float(env_reward)
"""

WRONG_ARITY = """
def reward(obs, action):
    return 0.0
"""


def run_one(label: str, src: str, expect_stage: str | None) -> bool:
    t0 = time.time()
    try:
        compile_reward(src)
        elapsed = time.time() - t0
        if expect_stage is None:
            print(f"  PASS  {label}: validated in {elapsed:.2f}s")
            return True
        print(f"  FAIL  {label}: expected RewardCompileError({expect_stage!r}), got success")
        return False
    except RewardCompileError as e:
        elapsed = time.time() - t0
        if expect_stage is None:
            print(f"  FAIL  {label}: expected success, got RewardCompileError({e.stage!r})")
            return False
        if e.stage != expect_stage:
            print(
                f"  FAIL  {label}: expected stage={expect_stage!r}, got stage={e.stage!r} after {elapsed:.2f}s"
            )
            return False
        print(f"  PASS  {label}: rejected as {e.stage!r} in {elapsed:.2f}s")
        return True


def main() -> int:
    cases: list[tuple[str, str, str | None | tuple[str, ...]]] = [
        ("valid reward source", VALID_SRC, None),
        ("syntax error", SYNTAX_BAD, "syntax-error"),
        ("import rejected", IMPORT_BAD, "ast-import-rejected"),
        # Either inner thread-based dry-run-timeout (caught in child) or
        # outer subprocess-timeout (parent kills child) is acceptable.
        # Both prove the timeout mechanism works.
        (
            "infinite loop (some timeout fires)",
            INFINITE_LOOP,
            ("subprocess-timeout", "dry-run-timeout"),
        ),
        ("wrong arity", WRONG_ARITY, "signature-arity"),
    ]
    passed = 0
    for label, src, stage in cases:
        # Accept tuple of acceptable stages
        if isinstance(stage, tuple):
            t0 = time.time()
            try:
                compile_reward(src)
                elapsed = time.time() - t0
                print(
                    f"  FAIL  {label}: expected one of {stage!r}, got success"
                )
            except RewardCompileError as e:
                elapsed = time.time() - t0
                if e.stage in stage:
                    print(f"  PASS  {label}: rejected as {e.stage!r} in {elapsed:.2f}s")
                    passed += 1
                else:
                    print(
                        f"  FAIL  {label}: expected one of {stage!r}, got {e.stage!r} after {elapsed:.2f}s"
                    )
        else:
            if run_one(label, src, stage):
                passed += 1
    print(f"\n{passed}/{len(cases)} sandbox smoke cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
