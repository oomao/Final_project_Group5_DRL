# `hermes_dqn` — DQN Baseline Package

Python package that implements the **DQN training loop** that Hermes-generated reward functions will plug into.

## Install

```bash
# Recommended: fresh venv
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# Install the package in editable mode
pip install -e .
```

### Windows + Box2D note

`gymnasium[box2d]` requires the `Box2D` C extension. If `pip install` fails with a Box2D build error:

1. Install **Visual C++ Build Tools** (Desktop development with C++) — <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
2. Install **swig**: `choco install swig` or download from <https://www.swig.org/>
3. Re-run `pip install -e .`

If still failing, install the pre-built wheel directly:

```bash
pip install box2d-py
```

## First training run

```bash
# Quick smoke test (~30 sec on CPU)
python -m hermes_dqn.training.train --episodes 10 --seed 42

# Full baseline run (~30-90 min on CPU, faster on GPU)
python -m hermes_dqn.training.train --episodes 1500 --seed 42
```

Outputs land in `runs/<YYYY-MM-DD_HH-MM-SS>/`:
- `config.json` — every hyperparameter + seed used
- `episodes.jsonl` — one JSON line per episode (return, length, ε, loss, wall-time)
- `model_final.pt` — Q-network weights after the last episode

## Score a completed run

```python
from hermes_dqn.training import FitnessEvaluator

report = FitnessEvaluator().evaluate("runs/2026-05-12_14-30-00/episodes.jsonl")
print(report)
# FitnessReport(converge_episode=1187, mean_reward_last100=228.4, success_rate=0.95, total_episodes=1500, seed=42)
```

## Plug in a custom reward function

```python
from hermes_dqn.env import make_env

def my_reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    # Penalize firing the main engine
    fuel_penalty = -0.1 if action == 2 else 0.0
    return env_reward + fuel_penalty

env = make_env(seed=42, reward_fn=my_reward)
```

Any function matching the 7-argument signature works — including code generated later by Hermes.

## Baseline runs

In-training fitness (computed on the reward source the agent saw):

| Date | Seed | Reward source | Converge ep | Mean (last 100) | Success rate | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-05-12 | 42 | env (native) | 399 | 262.79 | 0.95 | 24m47s |
| 2026-05-12 | 42 | llm (Gemma 4 31B) | 525 | 312.21 *(shaped)* | 0.85 *(shaped)* | 16m29s |

> Hardware (both runs): NVIDIA RTX 4090 / Windows 11 / Python 3.11 / torch 2.5.1+cu121.
> Mean and success rate for the llm row are computed on the *shaped* return the agent
> trained against (Gemma added shaping + terminal amplification on top of env reward),
> so the column is NOT directly comparable to the env row. See the apples-to-apples
> table below.

Apples-to-apples evaluation (greedy playback of the trained model on env-native reward, 100 unseen eval seeds 10000-10099):

| Reward source used in training | Mean env reward | Median | Success rate (≥200) | Crash rate (<0) | Mean ep length |
| --- | --- | --- | --- | --- | --- |
| env (native) | 162.72 | 226.43 | 53% | 14% | 265 |
| llm (Gemma 4 31B) | **207.72** | **238.02** | **78%** | **7%** | **419** |

The Gemma reward yields **+45 mean reward (+28%)**, **+25 pp success rate**, **halves crash rate**,
and trains **33% faster wall-clock**. First-shot LLM reward, no memory, no AST yet — this is
the EUREKA open-source replication thesis (Gemma replacing GPT-4) holding on seed 42. Multi-seed
verification queued for the `closed-loop-fitness` change per `experiments-protocol` spec.
