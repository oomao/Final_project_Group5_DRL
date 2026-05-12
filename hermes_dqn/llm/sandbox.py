"""Subprocess-isolated reward validation (L2 sandbox).

Why a subprocess: LLM-generated code may run an infinite loop inside a C
extension (numpy / etc.). Python threads cannot be killed, so the previous
threaded soft-timeout could leak hung threads. ``multiprocessing.Process``
can be hard-killed via ``terminate()``/``kill()``.

Why we re-compile in the parent after success: a compiled function cannot
be transferred across process boundaries (it carries closure/globals
references). Returning a callable would mean every training-loop call goes
through IPC, which is ~1000× slower. So the subprocess only **validates**;
the parent re-compiles the (now-trusted) source for in-process execution.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import sys
import traceback

from hermes_dqn.llm.compile import RewardCompileError

_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_MEMORY_MB = 512


def _validate_worker(src: str, q: "mp.queues.Queue", memory_mb: int | None) -> None:
    """Runs inside the child process. Reports outcome via ``q``."""
    try:
        # Linux-only memory cap. Windows: parent-side monitoring is best-effort
        # and only kicks in via psutil if available; without it we fall back to
        # timeout-only enforcement (documented in spec).
        if memory_mb is not None and sys.platform.startswith("linux"):
            import resource

            limit = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        from hermes_dqn.llm.compile import _validate_full

        _validate_full(src)
        q.put({"ok": True})
    except RewardCompileError as e:
        q.put(
            {
                "ok": False,
                "stage": e.stage,
                "message": e.message,
                "tb": e.tb,
            }
        )
    except BaseException as e:
        q.put(
            {
                "ok": False,
                "stage": "subprocess-uncaught",
                "message": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc(),
            }
        )


def validate_reward_in_subprocess(
    src: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    memory_mb: int | None = _DEFAULT_MEMORY_MB,
) -> None:
    """Validate LLM-generated reward source in a hard-kill subprocess.

    Returns ``None`` on success. Raises ``RewardCompileError`` on any failure
    mode: syntax / import / dunder rejection, signature arity, dry-run
    timeout, dry-run exception, OOM, or subprocess wall-time exceeded.
    """
    ctx = mp.get_context("spawn")
    q: "mp.queues.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_validate_worker,
        args=(src, q, memory_mb),
        daemon=True,
    )
    proc.start()
    result: dict | None = None
    try:
        try:
            result = q.get(timeout=timeout_s)
        except queue_module.Empty:
            result = None  # explicit: nothing received before deadline
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
                if proc.is_alive():
                    # Should be impossible; fatal at this point
                    raise RuntimeError(
                        f"sandbox subprocess pid={proc.pid} could not be killed"
                    )
        else:
            proc.join(timeout=0.5)

    if result is None:
        raise RewardCompileError(
            "subprocess-timeout",
            f"Reward validation exceeded {timeout_s:.1f}s wall-time and was terminated",
        )
    if not result["ok"]:
        raise RewardCompileError(
            result["stage"],
            result["message"],
            result.get("tb"),
        )
    return None
