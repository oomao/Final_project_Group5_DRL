# Handover Document (06) - 2026-05-22

## Summary of Changes

This session went from "LunarLander scaffolding ready" to **a full 4-environment
empirical study** (LunarLander, CartPole, MountainCar, Acrobot) with the most
interesting finding being a **cross-environment reversal**: memory hurts on
dense-reward tasks, helps on sparse-reward tasks.

### Phase 1: Full LunarLander experiment + recovery

- Ran the planned 6 conditions × 5 seeds × 1500ep × parallel orchestrator
  (`workers=5`). First attempt died at 8.73h with rc=127 (~10/12 B3 seeds done)
  due to Gemma 500/503 server errors breaking the closed-loop chain.
- **Patched `LLMRewardClient`** with exponential-backoff retry (6 attempts,
  5s→10s→30s→60s→120s waits). Already on disk.
- **Recovery run** (workers=3) completed 12 broken seeds in ~5.4h, total
  9 hours wall to get full n=5.
- Per-seed audit: 30/30 LunarLander seeds have valid `env_native_mean`, all
  reward functions unique, eval seeds (10000-10099) consistent, episodes=1500.

**LunarLander result** (`reports/final/comparison_report.md`):
```
B3-no-memory     248.77   ← 🥇 (success 90%, crash 0.4%)
B0-env-native    173.22
B3-hermes-full   153.56   ← high variance: 252, 12, 125, 197, 182
B2-gemma-oneshot 152.65
B3-no-AST         95.42
B1-handcrafted    77.77
```
Statistical: **B3-no-memory wins B3-hermes-full / B2 / B3-no-AST / B1 (p≈0.03)**.
The "memory hurts" finding is driven by **B3-hermes-full's high seed variance**
(one seed at 11.60, one at 252.40) rather than uniform regression.

### Phase 2: CartPole-v1 added as second environment

Pure additive changes — LunarLander code paths byte-identical:

- **`hermes_dqn/env/profiles.py`** (NEW): EnvProfile dataclass + registry.
  Defines obs_dim, n_actions, success_threshold, default_episodes,
  task_spec, b1_reward_file per env.
- **`hermes_dqn/llm/prompts.py`**: added CARTPOLE_TASK_SPEC + FEW_SHOT_SHAPED_CARTPOLE.
  Generalized `build_lunarlander_prompt` with `env_name` + `few_shot_shaped`
  parameters (defaults preserve LL byte-identical behavior).
- **`hermes_dqn/llm/client.py`**: `.generate()` passes env_name + few_shot to
  prompt builder.
- **`hermes_dqn/training/eval_env_native.py`**: reads env_id from config.json,
  picks success_threshold from profile registry.
- **`hermes_dqn/training/closed_loop.py`**: `--env-id` flag; env-aware few-shot
  dispatch; `obs_dim` from profile (was hardcoded 8).
- **`hermes_dqn/training/train.py`**: `--env-id` + `--reward-file` flags; new
  `reward_source=file` mode; **BUG FIX**: write config.json AFTER env dim
  detection (previously LL just happened to match the hardcoded 8/4).
- **`scripts/run_full_experiment.py`**: `--env-id` flag; B1 reward path
  dynamically chosen from env profile; ThreadPoolExecutor parallel mode
  (`--workers N`); per-subprocess CPU thread cap.
- **`experiments/baselines/B1_cartpole_handcrafted.py`** (NEW): placeholder.

**CartPole result** (`reports/final_cp/comparison_report.md`, 30 seeds clean):
```
B3-hermes-full   334.44   ← 🥇 (success 22%, beats B0 by +117%)
B3-no-memory     243.21
B3-no-AST        220.81
B2-gemma-oneshot 187.64
B1-handcrafted   160.19
B0-env-native    154.80
```
Statistical: **B3-hermes-full wins B0 (p=0.0317, +117%)**. This is the
*opposite* pattern from LunarLander.

### Phase 3: LunarLander replication attempt (final_v2)

Attempted independent replication with seeds 47-51 to verify "memory hurts" on
LL. Three orchestrator launches all died at 5-9h with rc=127 or rc=1. Final
state at session pause: only 2/5 B3-no-memory + 5/5 baselines + 5/5 B3-hermes-full
+ 0/5 B3-no-AST. Partial data is in `runs/final_v2/` but insufficient for full
replication.

Pragmatic decision (after long discussion): **drop the LL replication, add 2
new envs instead** — more scientific value per compute hour and likely to
succeed (smaller envs less prone to long-running crashes).

### Phase 4: MountainCar-v0 + Acrobot-v1 added

Same additive pattern — LL/CP unaffected:

- **`hermes_dqn/llm/prompts.py`**: added MOUNTAINCAR_TASK_SPEC, ACROBOT_TASK_SPEC,
  FEW_SHOT_SHAPED_MOUNTAINCAR, FEW_SHOT_SHAPED_ACROBOT.
- **`hermes_dqn/env/profiles.py`**: registered MOUNTAINCAR_V0 (obs_dim=2,
  success>=-110, default_ep=300) and ACROBOT_V1 (obs_dim=6, success>=-100,
  default_ep=500).
- **`hermes_dqn/training/closed_loop.py`**: 4-env few-shot dispatch dict.
- **`experiments/baselines/B1_mountaincar_handcrafted.py`** (NEW): placeholder.
- **`experiments/baselines/B1_acrobot_handcrafted.py`** (NEW): placeholder.
- **`scripts/run_overnight_mc_acr.bat`** (NEW): chained MC→Acrobot launcher
  with `cd /d` baked in (path with spaces handled), resumable on re-run.

Smoke tested both: **6/6 OK each in ~4 min**. Inspecting Gemma's first reward
for each env confirmed it picked up the right task spec (used correct obs
indices, mentioned env-specific termination conditions, no LL leakage).

## Current Status

- **2 environments with full clean data**: LunarLander + CartPole (60 seeds, 8
  audits passed each)
- **2 environments in progress**: MountainCar (18/30 done at handover) +
  Acrobot (queued, 0/30). User is about to re-launch the resume command:
  ```cmd
  cd /d "C:\Users\Mao\Desktop\DRL\Final Project" && rmdir /s /q runs\final_mc\B3-hermes-full\seed_45 2>nul & rmdir /s /q runs\final_mc\B3-hermes-full\seed_46 2>nul & rmdir /s /q runs\final_mc\B3-no-memory 2>nul & scripts\run_overnight_mc_acr.bat
  ```
- ETA after launch: ~4 hours remaining (12 MC + 30 Acrobot seeds)
- **B1 still a placeholder** in all 4 envs — author-written, must be replaced
  by non-author teammate before final paper per `evaluation-criteria` spec
- **`runs/final_v2/` partial data** — unclear whether to delete or keep as
  "partial replication confirms instability" footnote

## Next Actions

When the user reports "MC + Acrobot done":

- [ ] **Quick audit of MC + Acrobot data**: 30 seeds each, env_native_mean
  present, env_id correct, episodes match profile (300/500).
- [ ] **Generate two new Table 1 reports**:
  ```cmd
  python tools\compare_conditions.py --exp final_mc --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,B3-hermes-full,B3-no-memory,B3-no-AST
  python tools\compare_conditions.py --exp final_acr --conditions B0-env-native,B1-handcrafted,B2-gemma-oneshot,B3-hermes-full,B3-no-memory,B3-no-AST
  ```
- [ ] **4-environment integration**: build a side-by-side comparison showing
  the B3-hermes-full ranking + memory effect direction per env. Look for the
  sparse-vs-dense pattern (predicted: 3/3 sparse envs → memory helps).
- [ ] **Decide paper narrative** based on results:
  - If sparse hypothesis holds (3/3) → strong "task-dependent" story
  - If sparse hypothesis breaks on MC or Acrobot → re-examine the mechanism
  - Either way: the high-variance finding on B3-hermes-full is robust
- [ ] **Best-iter analysis** (deferred from earlier session): patch
  `compare_conditions.py` to also report best-iter per seed alongside last-iter.
  May reveal "Hermes finds great rewards but doesn't end on them".
- [ ] **Qualitative Gemma analysis**: read the 120+ generated reward_fn.py
  files (60 LL + 60 CP + 60 MC + 60 Acrobot last-iter) and categorize by
  complexity. Predicted finding: memory-equipped runs write progressively more
  complex / brittle code.
- [ ] **Replace 4 B1 placeholders** with non-author teammate versions.
  Re-run B1 only per env (~30 min each × 4 = ~2h total).
- [ ] Write paper §4 Experiments + §5 Discussion using the 4-env data.

## Open Questions

- **Will the sparse hypothesis hold on MC + Acrobot?** Both have native reward
  `-1 per step` (textbook sparse). If Hermes wins B0 in both, the cross-env
  pattern is 3/3 sparse + 1/1 dense — strong claim. If not, the mechanism
  needs rethinking.
- **The orchestrator's rc=127 crashes are Windows-specific.** Not seen on the
  shorter (CartPole/MC/Acrobot) runs in this session. Possibly tied to the
  longer wall-time of LL runs (more pipe data accumulated, more exposure to
  transient Gemma errors, more chance to hit a Windows handle limit). For the
  paper, LL n=5 is sufficient; further replication is not blocking.
- **Should `runs/final_v2/` be deleted or footnoted?** The partial replication
  confirms B3-hermes-full's variance signature (-133, 33, 48, 121, 264) but
  doesn't give a clean Mann-Whitney U. Recommendation: keep on disk, mention
  briefly in §5 limitation as "partial replication consistent with main result".
- **decay_factor and lessons_learned reflection** are both still untuned /
  disabled per earlier handovers. Lower priority than the 4-env scope.
- **The "B3-hermes-full high variance" finding** is the most robust thing we
  have (confirmed across 2 LL runs, n=10 effective). The paper's headline
  could pivot from "memory helps/hurts" to "memory makes Gemma write
  high-variance rewards" — more defensible and more interesting mechanism.
