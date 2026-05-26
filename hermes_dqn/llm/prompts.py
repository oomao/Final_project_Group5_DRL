"""Prompt templates for Gemma reward-function generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_dqn.memory.entry import MemoryEntry


MOUNTAINCAR_TASK_SPEC = """\
Environment: Gymnasium MountainCar-v0 (discrete actions).

Observation (numpy.ndarray, shape=(2,), dtype=float32):
  obs[0]  car position    (range [-1.2, 0.6]; goal is at >= 0.5, valley at -0.5)
  obs[1]  car velocity    (range [-0.07, 0.07])

Action (int in {0, 1, 2}):
  0  push left
  1  no push
  2  push right

Native env reward (passed in as env_reward):
  -1 per timestep until the goal is reached (sparse "time penalty").
  Episode terminates when position >= 0.5 (success) or 200 steps elapse (timeout).

Goal: drive the underpowered car up the right hill to position >= 0.5. The
engine cannot directly overpower gravity; the agent MUST rock back and forth
to build momentum. Standard "solved" criterion is mean episode return >= -110
over 100 trials (i.e., reach goal in average <110 steps).

Reward-shaping notes:
  - The native reward gives NO directional signal. Without shaping, DQN must
    discover the rock-back-and-forth strategy from sparse goal reward alone.
  - Productive shapings: bonus for height (position), bonus for absolute
    velocity (encourage motion), bonus for crossing the valley to the right.
  - WARNING: rewarding only position will trap the agent climbing-and-falling
    on the right slope. Pair position bonus with velocity bonus or a goal
    completion bonus.
"""


ACROBOT_TASK_SPEC = """\
Environment: Gymnasium Acrobot-v1 (discrete actions).

Observation (numpy.ndarray, shape=(6,), dtype=float32):
  obs[0]  cos(theta1)          (first joint angle, vertical-down = +1)
  obs[1]  sin(theta1)
  obs[2]  cos(theta2)          (second joint angle, relative to first link)
  obs[3]  sin(theta2)
  obs[4]  theta1 angular velocity   (range about [-4*pi, 4*pi])
  obs[5]  theta2 angular velocity   (range about [-9*pi, 9*pi])

Action (int in {0, 1, 2}):
  0  apply -1 torque to second joint
  1  apply  0 torque
  2  apply +1 torque

Native env reward (passed in as env_reward):
  -1 per timestep until the goal is reached (sparse "time penalty").
  Termination: -cos(theta1) - cos(theta1 + theta2) > 1.0 (i.e., the tip of
  the lower link swings above the horizontal pivot height). Step cap 500.

Goal: swing the double pendulum's tip above a target height by pumping
torque at the elbow joint. Like MountainCar, the actuator is underpowered;
the agent must pump energy in over multiple swings. Standard "solved"
criterion is mean episode return >= -100 over 100 trials.

Reward-shaping notes:
  - Native reward gives no signal until termination. Useful shapings:
    bonus for tip height (use -cos(theta1) - cos(theta1+theta2), higher is
    better), bonus for kinetic energy (encourage pumping).
  - Be careful with angular velocity bonuses — too large can encourage
    spinning in place without progress.
"""


CARTPOLE_TASK_SPEC = """\
Environment: Gymnasium CartPole-v1 (discrete actions).

Observation (numpy.ndarray, shape=(4,), dtype=float32):
  obs[0]  cart position             (range approximately [-4.8, 4.8])
  obs[1]  cart velocity
  obs[2]  pole angle in radians     (0 = vertical; episode ends at |angle| > 12 deg)
  obs[3]  pole angular velocity

Action (int in {0, 1}):
  0  push cart left
  1  push cart right

Native env reward (passed in as env_reward):
  +1 for every timestep the pole stays upright. No bonus for landing pose,
  no penalty for any action. The reward is sparse — every step is +1
  regardless of how well-centered or how slow the pole is moving.

Goal: keep the pole upright (|angle| < 12 deg) and the cart within bounds
(|x| < 2.4) for as long as possible. Episode ends at termination or at the
500-step cap (CartPole-v1). Standard "solved" criterion is mean episode
return >= 475 over 100 trials.

Reward-shaping notes for this env:
  - The native reward is informative only about "alive"; you may want to
    add denser signal (e.g. penalize large |angle|, large |x|, high
    angular velocity) to help the agent learn faster.
  - Do NOT directly return more than ~+10 per step — the buffer's TD-target
    scaling assumes returns of order +1 per step.
"""


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


FEW_SHOT_SHAPED_CARTPOLE = """\
EXAMPLE 2 (light shaping — penalize tilt + off-center, on top of env reward):
```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    angle = next_obs[2]
    tilt_penalty = abs(angle)            # radians; ~0 when vertical
    off_center_penalty = abs(x) / 2.4    # normalized by termination bound
    return float(env_reward) - 0.5 * tilt_penalty - 0.1 * off_center_penalty
```
"""


FEW_SHOT_SHAPED_MOUNTAINCAR = """\
EXAMPLE 2 (light shaping — reward height + speed, on top of env reward):
```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    pos = next_obs[0]
    vel = next_obs[1]
    # Bonus for being to the right of the valley (valley is at ~-0.5).
    height_bonus = (pos + 0.5) * 0.1
    # Bonus for absolute speed (encourage motion, helps escape valley).
    speed_bonus = abs(vel) * 5.0
    return float(env_reward) + height_bonus + speed_bonus
```
"""


FEW_SHOT_SHAPED_ACROBOT = """\
EXAMPLE 2 (light shaping — reward tip height, on top of env reward):
```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    cos_t1 = next_obs[0]
    cos_t12 = next_obs[2] * next_obs[0] - next_obs[3] * next_obs[1]
    # Tip height = -cos(theta1) - cos(theta1 + theta2); higher is better.
    # Threshold for termination is +1.0; native return is 0 at termination.
    tip_height = -cos_t1 - cos_t12
    height_bonus = 0.1 * tip_height
    return float(env_reward) + height_bonus
```
"""


_SYSTEM_PREAMBLE_TEMPLATE = """\
You are the reward-function author for a DQN agent learning Gymnasium's
{env_name}. Your job is to design a Python `reward` function that the
agent will train against. A well-designed reward function will let the DQN
converge faster and reach a higher final success rate than the native env
reward alone.

Hermes-DQN is a memory-augmented framework that iteratively improves the
reward function across multiple LLM iterations. In this iteration you will
produce one candidate; future iterations may build on it.
"""


# Backward-compat alias: any code that imported _SYSTEM_PREAMBLE got the
# LunarLander preamble; preserve that exact string.
_SYSTEM_PREAMBLE = _SYSTEM_PREAMBLE_TEMPLATE.format(env_name="LunarLander-v3")


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
    env_name: str = "LunarLander-v3",
    few_shot_shaped: str = FEW_SHOT_SHAPED,
) -> str:
    """Compose the full prompt for one Gemma generation attempt.

    When ``prior_attempts`` is non-empty, a "PRIOR HIGH-FITNESS ATTEMPTS" block
    is inserted after the task spec and before the few-shot examples. When
    ``prior_attempts`` is None or empty AND env_name == "LunarLander-v3" AND
    few_shot_shaped == FEW_SHOT_SHAPED, the output is byte-identical to the
    gemma-reward-generator-era prompt for the same other inputs.

    Pass ``env_name="CartPole-v1"`` + ``few_shot_shaped=FEW_SHOT_SHAPED_CARTPOLE``
    + ``task_spec=CARTPOLE_TASK_SPEC`` to retarget the prompt to CartPole.
    """
    system_preamble = _SYSTEM_PREAMBLE_TEMPLATE.format(env_name=env_name)
    parts: list[str] = [system_preamble, "TASK:", task_spec]

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
        parts.append(few_shot_shaped)

    if retry_context and not force_fallback:
        parts.append(
            "PRIOR ATTEMPT FAILED VALIDATION:\n"
            f"{retry_context}\n\n"
            "Produce a corrected version that fixes the issue above while\n"
            "keeping the same 7-arg signature and returning a float."
        )

    parts.append("YOUR RESPONSE (one Python code block, nothing else):")
    return "\n\n".join(parts)
