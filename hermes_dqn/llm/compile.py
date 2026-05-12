"""Sandboxed compile of an LLM-generated reward function.

The LLM is treated as a cooperative author that can make mistakes, NOT as an
adversary. The sandbox blocks accidental escapes (imports, dunder access,
non-whitelisted builtins) but does not attempt to resist a determined attacker.
"""

from __future__ import annotations

import ast
import inspect
import threading
import traceback
from typing import Any, Callable

import numpy as np


class RewardCompileError(Exception):
    """Raised by compile_reward when LLM source fails any validation stage."""

    def __init__(self, stage: str, message: str, tb: str | None = None):
        self.stage = stage
        self.message = message
        self.tb = tb
        super().__init__(f"[{stage}] {message}")


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "dict": dict,
    "list": list,
    "tuple": tuple,
    "pow": pow,
    "round": round,
    "isinstance": isinstance,
    "type": type,
    "print": print,
    "True": True,
    "False": False,
    "None": None,
}


_DRY_RUN_TIMEOUT_SEC = 0.1


def _reject_unsafe_ast(tree: ast.AST) -> None:
    """Walk the AST and raise RewardCompileError on disallowed constructs."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RewardCompileError(
                "ast-import-rejected",
                f"import statements are forbidden (line {node.lineno})",
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise RewardCompileError(
                "ast-dunder-rejected",
                f"attribute '{node.attr}' starting with '_' is forbidden (line {node.lineno})",
            )


def _extract_reward_callable(namespace: dict[str, Any]) -> Callable:
    if "reward" not in namespace:
        raise RewardCompileError(
            "missing-reward-symbol",
            "source did not define a top-level function named 'reward'",
        )
    fn = namespace["reward"]
    if not callable(fn):
        raise RewardCompileError(
            "reward-not-callable",
            f"'reward' is not callable (got {type(fn).__name__})",
        )
    sig = inspect.signature(fn)
    if len(sig.parameters) != 7:
        raise RewardCompileError(
            "signature-arity",
            f"reward must take exactly 7 args (got {len(sig.parameters)})",
        )
    return fn


def _dry_run(fn: Callable, timeout_sec: float = _DRY_RUN_TIMEOUT_SEC) -> Any:
    """Invoke the reward function once with synthetic args under a wall-time cap.

    On Windows we cannot use signal.alarm, so the timeout is implemented by
    running the call in a daemon thread and abandoning it after the deadline.
    The function won't be truly killed (Python threads aren't preemptible),
    but the dry-run is bounded and the result is detected as a timeout.
    """
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(8).astype(np.float32)
    next_obs = rng.standard_normal(8).astype(np.float32)
    action = 2
    env_reward = 0.5
    info: dict[str, Any] = {}

    result: list[Any] = [None]
    exc: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = fn(obs, action, next_obs, env_reward, False, False, info)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        raise RewardCompileError(
            "dry-run-timeout",
            f"reward call exceeded {timeout_sec * 1000:.0f}ms wall-time",
        )
    if exc[0] is not None:
        raise RewardCompileError(
            "dry-run-exception",
            f"reward raised {type(exc[0]).__name__}: {exc[0]}",
            tb="".join(traceback.format_exception(exc[0])),
        )
    return result[0]


def compile_reward(src: str) -> Callable:
    """Parse, sandbox-compile, and dry-run-validate an LLM-generated reward.

    Returns the validated callable. Any failure raises RewardCompileError
    whose ``stage`` field identifies which validation step rejected the source.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise RewardCompileError(
            "syntax-error",
            f"line {e.lineno}: {e.msg}",
            tb=traceback.format_exc(),
        ) from e

    _reject_unsafe_ast(tree)

    try:
        code = compile(tree, "<llm-reward>", "exec")
    except (SyntaxError, ValueError) as e:
        raise RewardCompileError(
            "compile-error",
            str(e),
            tb=traceback.format_exc(),
        ) from e

    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, "np": np}
    try:
        exec(code, namespace)
    except BaseException as e:
        raise RewardCompileError(
            "module-exec-error",
            f"top-level exec raised {type(e).__name__}: {e}",
            tb=traceback.format_exc(),
        ) from e

    fn = _extract_reward_callable(namespace)

    dry_result = _dry_run(fn)
    if not isinstance(dry_result, (int, float)) or isinstance(dry_result, bool):
        raise RewardCompileError(
            "dry-run-return-type",
            f"reward must return float or int (got {type(dry_result).__name__})",
        )

    return fn
