## Why

The Hermes-DQN architecture has full documentation but **zero implementation code**. Before any of the three subsystems (Hermes Agent memory, Gemma reward generator, AST/Buffer manager) can be evaluated, there must be a working DQN trainer that can (1) accept a swappable reward function and (2) emit fitness metrics. Without this load-bearing baseline, every later change has nothing to plug into — LLM-generated rewards cannot be scored, replay-buffer experiments cannot run, and end-to-end fitness loops cannot close. This change builds that foundation.

## What Changes

- Add a Python package `hermes_dqn/` with submodules `env/`, `agent/`, `training/`, `utils/`
- Implement a vanilla DQN agent (Q-network, target network, replay buffer, ε-greedy)
- Wrap Gymnasium `LunarLander-v3` with an injectable-reward interface
- Add `train.py` entry point: configurable episode count, logs reward curve and success rate to `runs/<timestamp>/`
- Define a `RewardFunction` plug-in contract (`Callable[[obs, action, next_obs, reward, terminated, truncated, info], float]`) so later Hermes-generated rewards drop in unchanged
- Define a `FitnessEvaluator` contract that consumes a training log and returns `{converge_episode, mean_reward_last100, success_rate}`
- Add `pyproject.toml` + `requirements.txt` pinning torch / gymnasium[box2d] / numpy
- Add a brief `hermes_dqn/README.md` covering install + first training run

## Capabilities

### New Capabilities
- `dqn-baseline`: Vanilla DQN training loop on LunarLander-v3 with reproducible logging
- `reward-plugin`: Injection contract that lets external reward functions (hand-written or LLM-generated) replace the env's default reward without touching agent or env code
- `fitness-evaluation`: Standardized scoring of a training run, producing the metrics later loops feed back into Hermes memory

### Modified Capabilities
- none (no existing specs touched)

## Impact

- New top-level Python package `hermes_dqn/` (currently the repo has no Python code)
- New `pyproject.toml`, `requirements.txt` at repo root
- New `runs/` directory (gitignored) for training artifacts
- `.gitignore` updated to exclude `runs/`, `__pycache__/`, `*.pt`, `.venv/`
- No changes to existing `.sh` scripts, `package.json`, or OpenSpec workflow
- External dependencies introduced: `torch`, `gymnasium[box2d]`, `numpy`, `tqdm` (pinned in requirements)
