"""LLMRewardClient: Gemma 4 31B reward-function generator with 3-retry sandbox.

The client wraps Google AI Studio's google-genai SDK. It:
1. Builds a prompt using ``hermes_dqn.llm.prompts.build_lunarlander_prompt``,
   optionally embedding prior high-fitness memory entries as in-context examples
2. Calls Gemma to produce Python source for a ``reward`` function
3. Validates the source via ``hermes_dqn.llm.compile.compile_reward`` (which
   internally uses the subprocess sandbox in ``hermes_dqn.llm.sandbox``)
4. On validation failure, re-prompts up to 3 total attempts (3rd forces fallback)
5. Optionally appends each attempt to a JSONL log for post-hoc inspection
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from google import genai

from hermes_dqn.llm.compile import RewardCompileError, compile_reward
from hermes_dqn.llm.prompts import (
    FEW_SHOT_SHAPED,
    LUNARLANDER_TASK_SPEC,
    build_lunarlander_prompt,
)

if TYPE_CHECKING:
    from hermes_dqn.memory.entry import MemoryEntry


_DEFAULT_MODEL = "gemma-4-31b-it"
_MAX_ATTEMPTS = 6  # bumped from 3 after observing 24% iter-failure rate from
# transient Gemma 500/503 server errors during the first `final` run.

# Backoff schedule (seconds) between API-error retries. Compile errors do NOT
# wait — the prompt is re-issued immediately with the error appended as context.
# Length must be >= _MAX_ATTEMPTS - 1.
_API_BACKOFF_S = [5, 10, 30, 60, 120]


@dataclass
class _Attempt:
    attempt: int
    prompt: str
    response: str
    error: str | None
    accepted: bool


class RewardGenerationError(Exception):
    """Raised when all _MAX_ATTEMPTS attempts produce unusable code."""

    def __init__(self, attempts: list[_Attempt]):
        self.attempts = attempts
        reasons = "; ".join(f"#{a.attempt}: {a.error}" for a in attempts if a.error)
        super().__init__(f"All {len(attempts)} attempts failed: {reasons}")


def _extract_code_block(response_text: str) -> str:
    py_match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if py_match:
        return py_match.group(1).strip() + "\n"
    any_match = re.search(r"```\s*\n?(.*?)```", response_text, re.DOTALL)
    if any_match:
        return any_match.group(1).strip() + "\n"
    return response_text.strip() + "\n"


def _write_attempts(path: Path, attempts: list[_Attempt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for a in attempts:
            fp.write(
                json.dumps(
                    {
                        "attempt": a.attempt,
                        "prompt": a.prompt,
                        "response": a.response,
                        "error": a.error,
                        "accepted": a.accepted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class LLMRewardClient:
    """Generate a validated reward-function source string from Gemma."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        resolved_key = api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and "
                "paste your Google AI Studio key (https://aistudio.google.com/app/apikey)."
            )
        self._api_key = resolved_key
        self.model = model or os.environ.get("GEMMA_MODEL") or _DEFAULT_MODEL
        self._client = genai.Client(api_key=resolved_key)

    def generate(
        self,
        task_spec: str = LUNARLANDER_TASK_SPEC,
        attempts_log_path: str | Path | None = None,
        memory: "list[MemoryEntry] | None" = None,
        env_name: str = "LunarLander-v3",
        few_shot_shaped: str = FEW_SHOT_SHAPED,
    ) -> str:
        """Return validated reward-function source code as a string.

        When ``memory`` is a non-empty list, a "PRIOR HIGH-FITNESS ATTEMPTS"
        section is added to the prompt summarizing those entries. ``memory=None``
        and ``memory=[]`` both produce the gemma-reward-generator-era behavior
        (no prior-attempts section).

        ``env_name`` + ``few_shot_shaped`` retarget the prompt to a non-LunarLander
        env (e.g. CartPole). Defaults preserve byte-identical LunarLander behavior.

        Raises ``RewardGenerationError`` after _MAX_ATTEMPTS failures.
        """
        attempts: list[_Attempt] = []
        retry_context: str | None = None
        prior_attempts = memory or None  # treat empty list as None

        for attempt_idx in range(1, _MAX_ATTEMPTS + 1):
            force_fallback = attempt_idx == _MAX_ATTEMPTS
            prompt = build_lunarlander_prompt(
                task_spec=task_spec,
                retry_context=retry_context,
                force_fallback=force_fallback,
                prior_attempts=prior_attempts,
                env_name=env_name,
                few_shot_shaped=few_shot_shaped,
            )

            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                response_text = response.text or ""
            except Exception as e:
                attempts.append(
                    _Attempt(
                        attempt=attempt_idx,
                        prompt=prompt,
                        response="",
                        error=f"api-call: {type(e).__name__}: {e}",
                        accepted=False,
                    )
                )
                retry_context = f"previous API call raised {type(e).__name__}: {e}"
                # Exponential backoff for transient server errors (500/503) and
                # rate limits (429). No-op on the final attempt.
                if attempt_idx < _MAX_ATTEMPTS:
                    wait_s = _API_BACKOFF_S[min(attempt_idx - 1, len(_API_BACKOFF_S) - 1)]
                    print(
                        f"[llm] API attempt {attempt_idx}/{_MAX_ATTEMPTS} failed "
                        f"({type(e).__name__}); sleeping {wait_s}s before retry...",
                        flush=True,
                    )
                    time.sleep(wait_s)
                continue

            source = _extract_code_block(response_text)

            try:
                compile_reward(source)
            except RewardCompileError as e:
                attempts.append(
                    _Attempt(
                        attempt=attempt_idx,
                        prompt=prompt,
                        response=response_text,
                        error=f"{e.stage}: {e.message}",
                        accepted=False,
                    )
                )
                retry_context = f"{e.stage}: {e.message}"
                continue

            attempts.append(
                _Attempt(
                    attempt=attempt_idx,
                    prompt=prompt,
                    response=response_text,
                    error=None,
                    accepted=True,
                )
            )
            if attempts_log_path is not None:
                _write_attempts(Path(attempts_log_path), attempts)
            return source

        if attempts_log_path is not None:
            _write_attempts(Path(attempts_log_path), attempts)
        raise RewardGenerationError(attempts)
