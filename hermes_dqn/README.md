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

Apples-to-apples evaluation (greedy playback on env-native reward, 100 unseen eval seeds 10000-10099). All on RTX 4090 / Windows 11 / Python 3.11 / torch 2.5.1+cu121.

| Run | Reward source | Memory state | Priors used | Mean env reward | Success ≥200 | Crash <0 | Mean ep length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_seed42` | env (native) | — | — | 162.72 | 53% | 14% | 265 |
| `gemma_seed42` | llm (Gemma 4 31B) | none | — | 207.72 | 78% | 7% | 419 |
| `gemma_mem_seed42` | llm + memory | hermes-sqlite-fts5 | `[]` (empty DB) | **235.21** | 80% | **3%** | 461 |
| `gemma_mem_seed43` | llm + memory | hermes-sqlite-fts5 | `[1]` (reads seed 42) | 224.53 | 78% | **3%** | 312 |

**What this shows (n=1, mechanism only — no statistical claim yet):**

- **EUREKA open-source replication still holds**: every llm-source run beats `baseline_seed42` (162.72) by a wide margin
- **Memory mechanism works end-to-end**: `gemma_mem_seed43` confirms `memory_priors_used=[1]` in its `config.json` — the second run read seed 42's reward+fitness as in-context prior, then wrote its own as entry id=2
- **Crash rate halves with memory** (7% → 3%); whether this is signal or noise needs 5-seed verification (queued for `closed-loop-fitness`)
- **n=1 caveat**: comparing 235 vs 224 vs 207 across single seeds is dominated by Gemma's stochastic output. The `experiments-protocol` spec mandates 5 seeds + Mann-Whitney U + bootstrap CI before any "memory helps" claim.

In-training shaped fitness (reward the agent *saw*, not directly comparable across reward sources):

| Run | Shaped mean (last 100) | Shaped success | Converge ep |
| --- | --- | --- | --- |
| `baseline_seed42` | 262.79 | 0.95 | 399 |
| `gemma_seed42` | 312.21 | 0.85 | 525 |
| `gemma_mem_seed42` | 221.72 | (varies) | (in `episodes.jsonl`) |
| `gemma_mem_seed43` | 164.88 | (varies) | (in `episodes.jsonl`) |
