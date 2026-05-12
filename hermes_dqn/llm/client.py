"""LLMRewardClient: Gemma 4 31B reward-function generator with 3-retry sandbox.

The client wraps Google AI Studio's google-genai SDK. It:
1. Builds a prompt using hermes_dqn.llm.prompts.build_lunarlander_prompt
2. Calls Gemma to produce Python source for a `reward` function
3. Validates the source via hermes_dqn.llm.compile.compile_reward
4. On validation failure, re-prompts up to 3 total attempts (3rd forces fallback)
5. Optionally appends each attempt to a JSONL log for post-hoc inspection
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from google import genai

from hermes_dqn.llm.compile import RewardCompileError, compile_reward
from hermes_dqn.llm.prompts import LUNARLANDER_TASK_SPEC, build_lunarlander_prompt


_DEFAULT_MODEL = "gemma-4-31b-it"
_MAX_ATTEMPTS = 3


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
    """Pull the first fenced Python code block out of a possibly-noisy response.

    Prefers ```python ... ```; falls back to the first ``` ... ``` of any flavor;
    falls back to the full response stripped if no fences exist.
    """
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
    ) -> str:
        """Return validated reward-function source code as a string.

        Raises RewardGenerationError after _MAX_ATTEMPTS failures.
        """
        attempts: list[_Attempt] = []
        retry_context: str | None = None

        for attempt_idx in range(1, _MAX_ATTEMPTS + 1):
            force_fallback = attempt_idx == _MAX_ATTEMPTS
            prompt = build_lunarlander_prompt(
                task_spec=task_spec,
                retry_context=retry_context,
                force_fallback=force_fallback,
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
