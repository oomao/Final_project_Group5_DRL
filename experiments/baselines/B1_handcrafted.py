"""B1-handcrafted: third-party hand-shaped reward for LunarLander-v3.

Per `openspec/specs/evaluation-criteria/spec.md` Requirement "Complete Baseline Set",
B1 must be authored by a *non-author* third party and frozen before evaluation.

STATUS: PLACEHOLDER (committed by primary author to unblock pipeline smoke tests).
ACTION BEFORE FINAL PAPER:
    1. A teammate who is NOT a co-author of the Gemma prompt pipeline rewrites this
       function from scratch using their own domain reasoning.
    2. Replace the entire body below their version.
    3. Commit with author trailer set to the teammate (Co-Authored-By: ...).
    4. Record the SHA-256 of the committed file in the experiment manifest.

Until step 1-4 are done, B1 results in any report MUST be footnoted as
"PLACEHOLDER pending third-party rewrite".

Design philosophy of this placeholder (purposely simple, no Gemma-style tricks):
    * Reward shaping uses ONLY textbook LunarLander signals: distance to landing
      pad, downward velocity, body tilt, leg contact.
    * No conditional branching on (terminated, env_reward) — a human reasonably
      writes additive shaping, not Gemma-style multi-branch logic.
    * No magic constants from prior experiments; weights are round numbers chosen
      "by intuition".

Observation indices (LunarLander-v3, gymnasium):
    obs[0] = x (horizontal position, ~0 over pad)
    obs[1] = y (vertical position, decreases as it descends)
    obs[2] = vx (horizontal velocity)
    obs[3] = vy (vertical velocity, negative = downward)
    obs[4] = angle (body tilt, ~0 upright)
    obs[5] = v_angle (angular velocity)
    obs[6] = left leg contact (0/1)
    obs[7] = right leg contact (0/1)
"""


def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]
    y = next_obs[1]
    vy = next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    # Start from the env's native shaped reward (LunarLander already gives
    # distance/velocity shaping plus large terminal bonuses/penalties).
    r = float(env_reward)

    # Small additional shaping a person would reasonably add:
    #  (a) prefer being centered over the pad
    r -= 0.1 * abs(x)
    #  (b) prefer being upright
    r -= 0.1 * abs(angle)
    #  (c) discourage fast downward velocity when low to the ground
    if y < 0.4 and vy < 0:
        r -= 0.1 * abs(vy)
    #  (d) bonus when both legs touch (encourages stable landing)
    if left_leg > 0.5 and right_leg > 0.5:
        r += 1.0

    return float(r)
