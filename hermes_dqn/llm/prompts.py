"""Prompt templates for Gemma reward-function generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_dqn.memory.entry import MemoryEntry


LUNARLANDER_TASK_SPEC = """\
Environment: Gymnasium LunarLander-v3 (discrete actions).

Observation (numpy.ndarray, shape=(8,), dtype=float32):
  obs[0]  x position           (negative=left, positive=right, landing pad at 0)
  obs[1]  y position           (positive=up, landing pad at 0)
  obs[2]  x velocity
  obs[3]  y velocity
  obs[4]  angle in radians     (0 = upright, positive = tilted right)
  obs[5]  angular velocity
  obs[6]  left leg contact     (0.0 or 1.0)
  obs[7]  right leg contact    (0.0 or 1.0)

Action (int in {0, 1, 2, 3}):
  0  do nothing
  1  fire left orientation engine
  2  fire main engine (boost up)
  3  fire right orientation engine

Native env reward (passed in as env_reward):
  +100 to +140 for soft landing between the flags
  +10 per leg contact
  -100 for crashing
  -0.3 per frame firing the main engine
  -0.03 per frame firing a side engine
  Shaped term that rewards moving toward the pad with zero velocity, upright.

Goal: land softly between the flags at (x=0, y=0) with low velocity, upright,
both legs in contact, using as little fuel as possible. Episode ends when the
lander lands, crashes, flies off-screen, or hits the 1000-step cap.
"""


_RESPONSE_FORMAT = """\
RESPONSE FORMAT:
Reply with exactly one fenced Python code block containing a top-level
function named `reward` with this exact 7-argument signature:

```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    ...
    return float_value
```

Rules:
- Return a single float. Do NOT return None, np.ndarray, or any non-numeric.
- You MUST NOT use any `import` statement. `np` (numpy) is already in scope.
- You MUST NOT access any name starting with `_` (no dunder escape).
- Available built-ins: abs, min, max, sum, len, range, float, int, bool, dict, list, tuple, pow, round, isinstance, type, print, True, False, None.
- The function MUST complete in well under 100 ms per call (the dry-run will reject slow code).
- Do not include explanatory prose outside the code block.
"""


FEW_SHOT_PASSTHROUGH = """\
EXAMPLE 1 (passthrough — always valid):
```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    return float(env_reward)
```
"""


FEW_SHOT_SHAPED = """\
EXAMPLE 2 (light shaping — upright + centered bonus on top of env reward):
```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    angle = next_obs[4]
    upright_bonus = max(0.0, 1.0 - abs(angle))
    center_bonus = max(0.0, 1.0 - abs(x))
    return float(env_reward) + 0.1 * upright_bonus + 0.1 * center_bonus
```
"""


_SYSTEM_PREAMBLE = """\
You are the reward-function author for a DQN agent learning Gymnasium's
LunarLander-v3. Your job is to design a Python `reward` function that the
agent will train against. A well-designed reward function will let the DQN
converge faster and reach a higher final success rate than the native env
reward alone.

Hermes-DQN is a memory-augmented framework that iteratively improves the
reward function across multiple LLM iterations. In this iteration you will
produce one candidate; future iterations may build on it.
"""


def _format_prior_attempts(prior_attempts: "list[MemoryEntry]") -> str:
    """Render the PRIOR HIGH-FITNESS ATTEMPTS block. Caller ensures list is non-empty."""
    parts: list[str] = [
        "PRIOR HIGH-FITNESS ATTEMPTS (use these as inspiration, don't copy verbatim):",
    ]
    for i, e in enumerate(prior_attempts):
        if e.env_native_mean is not None:
            fitness_line = (
                f"env_native_mean={e.env_native_mean:.2f}, success_rate={e.success_rate:.2f}"
            )
        else:
            fitness_line = (
                f"mean_reward_last100={e.mean_reward_last100:.2f} (shaped, no apples-to-apples eval available), "
                f"success_rate={e.success_rate:.2f}"
            )
        attempt_label = chr(ord("A") + i)
        parts.append(
            f"\nAttempt {attempt_label} ({fitness_line}):\n"
            f"```python\n{e.reward_code.rstrip()}\n```"
        )
        if e.lessons_learned:
            parts.append(f"Lessons: {e.lessons_learned}")
    return "\n".join(parts)


def build_lunarlander_prompt(
    task_spec: str = LUNARLANDER_TASK_SPEC,
    retry_context: str | None = None,
    force_fallback: bool = False,
    prior_attempts: "list[MemoryEntry] | None" = None,
) -> str:
    """Compose the full prompt for one Gemma generation attempt.

    When ``prior_attempts`` is non-empty, a "PRIOR HIGH-FITNESS ATTEMPTS" block
    is inserted after the task spec and before the few-shot examples. When
    ``prior_attempts`` is None or empty, the output is byte-identical to the
    gemma-reward-generator-era prompt for the same other inputs.
    """
    parts: list[str] = [_SYSTEM_PREAMBLE, "TASK:", task_spec]

    if prior_attempts:
        parts.append(_format_prior_attempts(prior_attempts))

    parts.append(_RESPONSE_FORMAT)

    if force_fallback:
        parts.append(
            "FALLBACK INSTRUCTION:\n"
            "Two previous attempts failed validation. To unblock the training\n"
            "run, emit the simplest valid reward exactly:\n\n"
            "```python\n"
            "def reward(obs, action, next_obs, env_reward, terminated, truncated, info):\n"
            "    return float(env_reward)\n"
            "```\n"
        )
    else:
        parts.append(FEW_SHOT_PASSTHROUGH)
        parts.append(FEW_SHOT_SHAPED)

    if retry_context and not force_fallback:
        parts.append(
            "PRIOR ATTEMPT FAILED VALIDATION:\n"
            f"{retry_context}\n\n"
            "Produce a corrected version that fixes the issue above while\n"
            "keeping the same 7-arg signature and returning a float."
        )

    parts.append("YOUR RESPONSE (one Python code block, nothing else):")
    return "\n\n".join(parts)
