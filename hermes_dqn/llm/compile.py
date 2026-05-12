"""Sandboxed compile of an LLM-generated reward function.

LLM is a cooperative author that makes mistakes, not an adversary. Multi-layer
protection:

- L1 (this module): AST blacklist (imports, dunder access) + restricted-globals exec
  with a builtins whitelist + signature-arity check + dry-run with return-type check
- L2 (``hermes_dqn.llm.sandbox``): the validation pipeline above is run in a hard-
  killable ``multiprocessing.Process`` so dry-run hangs (esp. in C extensions)
  cannot leak threads back into the training process
- L3 (planned, see ``reward-sandbox-isolation`` change): container-level isolation
  for file-system and network egress; out of scope here

This file exposes ``compile_reward`` (public; subprocess-validated by default)
and three internal helpers used by ``sandbox.py`` and the legacy debug path:

- ``_ast_check_and_exec``: parse + reject + exec + extract callable. NO dry-run.
- ``_dry_run``: invoke a callable once on a synthetic transition under a soft
  100 ms thread-based cap. Used inside the subprocess (where the outer hard
  timeout catches anything this misses).
- ``_validate_full``: the pipeline run inside the subprocess by sandbox.py.
"""

from __future__ import annotations

import ast
import inspect
import sys
import threading
import traceback
from typing import Any, Callable

import numpy as np


class RewardCompileError(Exception):
    """Raised by validation/compile when LLM source fails any stage."""

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


def _ast_check_and_exec(src: str) -> Callable:
    """Parse, reject unsafe, exec with restricted globals, return callable. NO dry-run."""
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

    return _extract_reward_callable(namespace)


def _dry_run(fn: Callable, timeout_sec: float = _DRY_RUN_TIMEOUT_SEC) -> Any:
    """Invoke fn once on a synthetic transition under a soft thread-based cap.

    Used inside the subprocess; the parent's hard-kill timeout covers cases
    where this thread join times out (Python threads cannot truly be killed).
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


def _validate_full(src: str) -> None:
    """Full validation pipeline. Designed to be invoked inside sandbox.py's subprocess.

    Raises RewardCompileError on any failure with a populated ``stage``.
    Returns None on success.
    """
    fn = _ast_check_and_exec(src)
    dry_result = _dry_run(fn)
    # Accept Python int/float AND numpy scalar types (numpy float32 etc. fail
    # isinstance(float) on most systems); reject bool (subtype of int).
    if isinstance(dry_result, bool):
        raise RewardCompileError(
            "dry-run-return-type",
            f"reward must return float or int (got bool)",
        )
    try:
        float(dry_result)
    except (TypeError, ValueError):
        raise RewardCompileError(
            "dry-run-return-type",
            f"reward must return float or int (got {type(dry_result).__name__})",
        )


_UNSAFE_INLINE_WARNED = False


def compile_reward(src: str, *, _unsafe_inline: bool = False) -> Callable:
    """Validate and return a callable for the LLM-generated reward.

    Default path uses ``hermes_dqn.llm.sandbox.validate_reward_in_subprocess``
    to run the validation pipeline in a hard-killable subprocess. After
    validation passes, the source is re-compiled inline in the calling process
    (no dry-run; already validated) so training-loop calls do not pay any IPC
    cost. ``_unsafe_inline=True`` skips the subprocess hop — only for debugging
    when you need traceback line numbers aligned with the LLM-emitted source.
    """
    if _unsafe_inline:
        global _UNSAFE_INLINE_WARNED
        if not _UNSAFE_INLINE_WARNED:
            sys.stderr.write(
                "WARN: compile_reward sandbox bypassed (--unsafe-inline-compile); debug only.\n"
            )
            _UNSAFE_INLINE_WARNED = True
        _validate_full(src)
        return _ast_check_and_exec(src)

    # Local import avoids a cycle: sandbox.py imports RewardCompileError from here.
    from hermes_dqn.llm.sandbox import validate_reward_in_subprocess

    validate_reward_in_subprocess(src)
    return _ast_check_and_exec(src)
