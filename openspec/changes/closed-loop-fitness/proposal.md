## Why

四個前置 change 各自把一塊磚做好,但**還沒接在一起**:

- `bootstrap-dqn-baseline`:DQN 訓練底盤 + Fitness 評估
- `gemma-reward-generator`:Gemma 寫一份 reward
- `hermes-memory-layer`:Hermes 4 層記憶第一層 + L2 sandbox
- `ast-buffer-manager`:AST diff + buffer 處理策略

每個 change 都已 strict-valid 並有 n=1 證據。但**整套系統的核心命題「Gemma + 記憶 + AST/Buffer 三者協同,讓 reward 寫得越來越好」目前沒被驗證過** ── 因為我們只跑過單輪 reward 生成。

本 change 是收尾,做兩件事:

1. **多輪閉環引擎**:把四個元件串成「**Hermes 翻筆記 → Gemma 寫 reward → AST 看差異 → Buffer 套策略 → DQN 訓練 → fitness 回饋 → 寫進筆記 → 下一輪**」7 步迴圈,可重複 N=5 iterations 跨多個 seed/condition
2. **統計比較工具**:`experiments-protocol` / `evaluation-criteria` 兩份治理 spec 規定的 Mann-Whitney U + 5000-bootstrap CI + win-3-條件,做成命令列工具,任何人跑完正式 6-condition × 5-seed run 都能直接出論文 Table 1

本 change 完成後,專案進入「**可以跑正式實驗、出論文數字**」的狀態。完整的 60+ GPU-hour 統計 run **不在 apply 階段執行**(那是 demo 前的實驗週),但 pilot smoke(1 seed × 3 iter ≈ 75 min)會跑通確認閉環無 bug。

## What Changes

### A. 多輪閉環引擎 — `hermes_dqn/training/closed_loop.py`

公開 `run_closed_loop(exp_name, condition_id, seed, n_iterations=5, dqn_episodes=1500, memory_db, decay_factor=0.5) -> ClosedLoopSummary` 函式,每個 iteration 做 7 件事:

1. **Hermes prior**:`MemoryStore(memory_db).top_k_by_fitness(k)` 拉前 K 筆高 fitness reward
2. **Gemma generate**:`LLMRewardClient.generate(memory=priors)` 寫新 reward(走既有 L2 sandbox)
3. **AST diff**:`diff_rewards(prev_iter_reward_src, new_src)` 算差異(iter 1 跳過)
4. **Buffer policy**:`apply_policy(inherited_buffer, decide_policy(diff), decay_factor)`(iter 1 用空 buffer)
5. **DQN train**:用 inherited buffer + 新 reward 跑 1500 ep
6. **Env-native eval**:既有 `evaluate_on_env_native(run_dir, n=100)`
7. **Persist**:`memory.write(MemoryEntry)` + `buffer.save("buffer.npz")` 留給下輪

CLI 入口:`python -m hermes_dqn.training.closed_loop --exp-name <X> --condition-id <Y> --seed <S> --iterations 5`

Run 目錄階層(對齊 `experiments-protocol R6 / R7`):
```
runs/<exp_name>/<condition_id>/seed_<NN>/
├── summary.json                # iter-level fitness 序列、wall_time、AST diff 連續紀錄
└── iter_<II>/                  # 每個 iteration 一個既有的 single-run 結構
    ├── config.json
    ├── episodes.jsonl
    ├── reward_fn.py
    ├── model_final.pt
    ├── llm_attempts.jsonl
    └── buffer.npz              # 給下個 iteration load
```

### B. 統計比較工具 — `tools/compare_conditions.py`

CLI:`python tools/compare_conditions.py --exp <X> --conditions B0,B1,B2,B3,B3-no-memory,B3-no-AST`

對每對 condition 做:
- Mann-Whitney U test(雙尾 α=0.05)
- 5000-bootstrap 95% CI for mean env_native_mean
- Win 三條件判定(p < 0.05 AND 平均差 ≥ 10% AND CI 不重疊)

輸出:
- `reports/<exp>/comparison_report.md`:每對 condition 的 p-value 矩陣 + 條件總覽表
- `reports/<exp>/figures/training_curves.png`:每 condition 一條 reward-over-episode 曲線 + shaded CI(對齊 `evaluation-criteria R5`)
- `reports/<exp>/figures/iteration_fitness.png`:每 condition 的 fitness-over-LLM-iteration 折線

### C. Pilot smoke(apply 時必跑)

1 seed × 3 iterations,~75 min on 4090。驗收:
- iter 2 與 iter 3 的 LLM prompt 確實含 iter 1/2 的 reward 作為 prior(已被 hermes-memory-layer 驗過,這裡確認**跨 iter** 而不只跨 run 仍成立)
- iter 2 / 3 的 buffer 真的從前一輪 load,且 `apply_policy` 確實依 diff 做了 KEEP/DECAY/CLEAR
- iter 1/2/3 的 env_native_mean 列出來,觀察是否單調上升(不要求 ── 只跑 pilot,結論留給正式 5-seed run)

## Capabilities

### New Capabilities

- `closed-loop-engine`:多輪迭代訓練主迴圈,串接 memory / LLM / sandbox / AST diff / buffer policy / 訓練 / fitness eval / persist
- `condition-comparison`:基於 `evaluation-criteria` 治理 spec 的統計比較工具(Mann-Whitney + bootstrap CI + 三條件 Win 判定 + 報告生成)

### Modified Capabilities

- none(本 change 完全是消費者:讀 memory-store / llm-reward-client / ast-buffer-manager 的既有 API,寫到 runs/ + reports/。所有底層介面已穩定)

## Impact

- 新增檔案:
  - `hermes_dqn/training/closed_loop.py`(主迴圈 + CLI)
  - `hermes_dqn/training/summary.py`(`ClosedLoopSummary` dataclass + JSON serializer)
  - `tools/compare_conditions.py`(統計比較 CLI)
  - `reports/`(gitignored,如同 `runs/`)
- 修改檔案:
  - `.gitignore`:加 `reports/` 排除
  - `hermes_dqn/training/__init__.py`:re-export `run_closed_loop`
  - `hermes_dqn/__init__.py`:版本號 `0.2.0` → `0.3.0`
- 新增 Python 相依:`scipy~=1.14`(for `scipy.stats.mannwhitneyu`)
- 不破壞既有 `train.py`:`closed_loop.py` 是新 CLI,既有 `python -m hermes_dqn.training.train ...` 完全照舊運作
- 引用的 spec scenarios:
  - `establish-project-lifecycle-spec / experiments-protocol`:R1(5 seeds default `[42-46]`)、R2(condition 三元組)、R3(1500 ep 固定無 early-stop)、R4(5 iter)、R5(reward_fn.py + SHA-256)、R6(3-level hierarchy)、R7(smoke/pilot/full 三 size)
  - `establish-project-lifecycle-spec / evaluation-criteria`:R1(6 baselines)、R2(Mann-Whitney + bootstrap)、R3(Win 三條件)、R4(primary = converge_episode)、R5(training curves 視覺化)、R6(run 通過條件)、R7(outliers 全 report)、R8(compute cost 必呈報)
  - `bootstrap-dqn-baseline / fitness-evaluation`:`FitnessEvaluator.evaluate(...)` 既有 API
  - `gemma-reward-generator / llm-reward-client` MODIFIED by `hermes-memory-layer`:`generate(memory=...)`
  - `hermes-memory-layer / memory-store + memory-llm-integration`:`MemoryStore.write / top_k_by_fitness`
  - `ast-buffer-manager / ast-buffer-manager`:`diff_rewards / decide_policy / apply_policy` + ReplayBuffer save/load/decay/clear
