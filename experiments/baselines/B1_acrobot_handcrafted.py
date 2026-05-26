"""B1-handcrafted (Acrobot-v1): third-party hand-shaped reward.

STATUS: PLACEHOLDER (author-written stand-in; replace with non-author teammate
version before final paper per `evaluation-criteria` Requirement
"Complete Baseline Set"). See `B1_handcrafted.py` for the 4-step replacement
protocol.

Design philosophy (textbook Acrobot shaping):
    * Native reward is -1 per step (sparse).
    * Add bonus for tip height = -cos(theta1) - cos(theta1+theta2).
      The termination condition is tip_height > 1.0, so this directly
      rewards "getting closer to swinging up".
    * Keep magnitude small so native -1 still drives "do it fast".

Observation indices (Acrobot-v1):
    obs[0] = cos(theta1)
    obs[1] = sin(theta1)
    obs[2] = cos(theta2)
    obs[3] = sin(theta2)
    obs[4] = theta1 angular velocity
    obs[5] = theta2 angular velocity
"""


def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    cos_t1 = next_obs[0]
    sin_t1 = next_obs[1]
    cos_t2 = next_obs[2]
    sin_t2 = next_obs[3]

    # cos(t1 + t2) = cos(t1)*cos(t2) - sin(t1)*sin(t2)
    cos_t1_plus_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2

    # Tip height (negative-cos sum). Range ~ [-2, +2]. Termination at > 1.0.
    tip_height = -cos_t1 - cos_t1_plus_t2

    # Native time penalty
    r = float(env_reward)

    # Height bonus: 0.1 weight keeps it under native -1.0 dominance,
    # so agent still prefers shorter episodes.
    r += 0.1 * tip_height

    return float(r)
