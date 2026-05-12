## Context

`bootstrap-dqn-baseline` 證明 DQN baseline 可收斂。
`gemma-reward-generator` 證明 Gemma 寫的 reward 比 env-native 好(seed 42 +28%)。
`hermes-memory-layer` 證明 priors 可被跨 run 累積。
`ast-buffer-manager` 提供 AST diff + buffer 政策的純函式庫。

但**這四個 change 各自只接到 train.py 的單發訓練流程**,還沒有「**一輪 iteration 結束之後,下一輪自動把 buffer 與 reward 串起來繼續做**」這條主迴圈。本 change 補上這條主迴圈 + 治理 spec 規定的統計工具。

關鍵設計選擇:
- 主迴圈是 **library 函式** 而非單純 CLI,讓正式 5-seed run 可以從 `experiments_runner.py` 之類的腳本批次呼叫
- 完全**不修改 `train.py`** ── closed_loop 自己處理 buffer save/load + iteration handoff
- 跨 iteration 的 buffer **存到 `iter_<II>/buffer.npz`**,下一輪 load(已由 `ast-buffer-manager` 提供 API)
- 統計工具與閉環引擎**完全解耦**:閉環只負責把 run 結果寫出來,統計工具獨立讀 + 算

## Goals / Non-Goals

**Goals:**
- `run_closed_loop(exp, cond, seed, N=5)` 完整跑完 N 個 iteration,寫齊所有 artifact,訊號清楚地告知失敗 iter
- run 目錄階層完全符合 `experiments-protocol R6`(`runs/<exp>/<cond>/seed_<NN>/iter_<II>/`)
- pilot smoke(3 iter)≤ 90 分鐘 wall-time 完成
- `compare_conditions.py` 給定 ≥ 2 condition 即可產出 markdown 報告 + 至少 1 張圖
- 統計檢定的 win 判定完全符合 `evaluation-criteria R3`(p<0.05 AND ≥10% AND CI 不重疊)
- 對既有 `train.py` 完全 backward-compat ── 任何單發訓練命令都還能跑

**Non-Goals:**
- 自動化跑滿 6 conditions × 5 seeds(那是「實驗週」的事,本 change 提供基礎建設不執行)
- B1 hand-shaped reward 的撰寫(留給第三方作者)
- 跨多台機器分散運算(本 change 假設單機 4090)
- 跨任務 transfer(LunarLander → CartPole)
- Hermes 其他三層記憶(Short Context / Working / Procedural)── 留給後續 change
- Live progress dashboard(JSONL log 加 stdout tqdm 就夠用)

## Decisions

### A. closed_loop.py 結構

```python
def run_closed_loop(
    exp_name: str,
    condition_id: str,
    seed: int,
    n_iterations: int = 5,
    dqn_episodes: int = 1500,
    memory_db: str | Path | None = None,
    decay_factor: float = 0.5,
    eval_n_episodes: int = 100,
    out_root: str | Path = "runs",
) -> ClosedLoopSummary:
    """Runs N iterations of the Hermes-DQN closed loop for one (condition, seed)."""
```

實作上每 iteration 構造一個 `TrainConfig`,呼叫既有 `train.py::train()` 函式(programmatic,不 fork subprocess)。closed_loop 自己處理:
- iter 0 之前:創建 `seed_dir`,空 buffer
- 每 iter 開始:把 inherited buffer 注入到 agent.replay_buffer
- 每 iter 完成:存 buffer,讀 fitness 寫 memory,把 reward_src 留給下一輪做 AST diff

**Alternative considered**:每 iter 跑 subprocess。被否決,subprocess 啟動成本 + memory db 競爭風險;in-process 直接呼叫 `train()` 函式更乾淨。

### B. Buffer 的 inter-iteration 流動

```
iter 1:
  reward_fn = Gemma.generate(memory=[])
  agent.replay_buffer = ReplayBuffer(seed=seed, ...)     # 空
  train()
  agent.replay_buffer.save(iter_01/buffer.npz)
  prev_reward_src = reward_fn_source

iter 2:
  reward_fn = Gemma.generate(memory=top_k_priors)
  diff = diff_rewards(prev_reward_src, reward_fn_source)
  action = decide_policy(diff)                            # KEEP / DECAY / CLEAR
  agent.replay_buffer = ReplayBuffer(seed=seed, ...)
  agent.replay_buffer.load(iter_01/buffer.npz)            # 載入前一輪
  apply_policy(agent.replay_buffer, action, decay_factor) # 套政策
  train()
  agent.replay_buffer.save(iter_02/buffer.npz)
  prev_reward_src = reward_fn_source

iter 3, 4, 5: 同 iter 2
```

**Decay factor 0.5**:即 `STRUCTURAL_DIFF` 時舊樣本權重打對折。實際值需要 ablation,本 change 預設 0.5 並允許 CLI 覆寫。

### C. train.py 的整合方式

要在 closed_loop 中呼叫 `train()` 並能注入 inherited buffer。當前 `train.py::train(config)` 自己 new 一個 `DQNAgent`,而 DQNAgent 內部建 ReplayBuffer。沒辦法外部塞 buffer。

**選擇 A**:重構 `train.py::train()` 讓它接受 `pre_loaded_buffer: ReplayBuffer | None = None` 參數。如果有給,agent 用它取代自己 new 的。
- **Pros**:相對乾淨,訓練主迴圈不變
- **Cons**:小幅修改 `train.py` 的簽章,需要更新 `bootstrap-dqn-baseline` 的 fitness-evaluation spec? 答:沒影響,簽章是 backward-compat(新參數預設 None)

**選擇 B**:closed_loop 自己整段抄一份 train loop,不呼叫 train.py。
- **Pros**:零修改 train.py
- **Cons**:Code 重複,將來 train.py 改了 closed_loop 要同步

**決議:選擇 A**(改 train.py 加 optional 參數)。代價低,維護性高。Backward-compat 保證:沒傳 `pre_loaded_buffer` 時行為 byte-identical。

### D. `summary.json` schema

```json
{
  "exp_name": "pilot",
  "condition_id": "B3-pilot",
  "seed": 42,
  "n_iterations": 3,
  "iterations": [
    {
      "iter": 1,
      "reward_fn_sha256": "...",
      "memory_priors_used": [],
      "diff_from_prev": null,
      "buffer_action": null,
      "env_native_mean": 207.7,
      "env_native_success": 0.78,
      "env_native_crash_rate": 0.07,
      "shaped_mean_last100": 312.2,
      "converge_episode": 525,
      "wall_time_s": 990.5
    },
    {
      "iter": 2,
      ...
      "diff_from_prev": {"kind": "STRUCTURAL_DIFF", "similarity": 0.82},
      "buffer_action": "decay",
      ...
    }
  ],
  "total_wall_time_s": 4500.0
}
```

### E. compare_conditions.py 演算法

```
1. Discover: glob runs/<exp>/<cond>/seed_*/iter_*/config.json
2. For each condition: take LAST iter per seed (or specified --iter)
3. Build {condition: [env_native_mean per seed]}
4. For each pair (A, B):
   - Mann-Whitney U two-sided p
   - bootstrap 95% CI: 5000 resamples of mean difference
   - Compute means_diff_pct
   - Apply 3-condition win: p<0.05 AND diff_pct>=10 AND CIs don't overlap
   - Classify: A wins / B wins / inconclusive
5. Generate report.md with:
   - Summary table (one row per condition: n, mean ± CI, success, crash, wall_time)
   - Pairwise matrix
   - Outlier disclosure (n_divergent per condition)
6. Generate figures:
   - training_curves.png: x=episode, y=return, one line per condition, shaded 95% CI across seeds
   - iteration_fitness.png: x=iter, y=env_native_mean, one line per (condition,seed)
```

**Alternative considered**:用 t-test。被否決,n=5 不能假設常態,`evaluation-criteria R2` 也明確規定 Mann-Whitney。

### F. Library + CLI 分離

`run_closed_loop` 是 library 函式,**不 print 任何 user-facing 字眼**(只用 logging)。CLI 部分:

```python
def main():
    args = _build_argparser().parse_args()
    summary = run_closed_loop(
        exp_name=args.exp_name,
        condition_id=args.condition_id,
        ...
    )
    print(f"\n[OK] Closed loop {summary.n_iterations} iter done.")
    print(f"Summary: {summary.iterations[-1].env_native_mean:.2f} env_native_mean at last iter")
```

這樣後續 `experiments_runner.py`(本 change 不寫,留給「實驗週」)可以 import library 而不撞 stdout。

### G. 失敗 iter 的處理

如果某 iter 的 Gemma 三次重試都失敗,或訓練 crash:
- 標記 `iterations[i].status = "failed"`,寫 error message 進 summary
- **不中斷後續 iteration**(下一個 iter 用空 buffer + 不讀本 iter 的 prior)
- summary 仍寫出，方便 post-mortem

理由:long-running 實驗單一 iter 失敗不應該丟掉前面的成果。

**Alternative considered**:整個 closed_loop 馬上 exit non-zero。被否決,失敗 retry 應該是 batch runner 的責任(屬實驗週的工作),不是 closed_loop 自己。

### H. scipy 相依

`compare_conditions.py` 用 `scipy.stats.mannwhitneyu`。其他選項:
- 自己實作 Mann-Whitney(複雜,易錯)
- 用 `statsmodels`(更大相依)

選 `scipy` ── 標準科學計算套件,版本 1.14+ 穩定。

## Risks / Trade-offs

- **train.py 加 optional 參數 `pre_loaded_buffer`**:小幅 API change。**Mitigation**:預設 None,既有 caller 不受影響;backward-compat smoke 跑一次 byte-identical
- **closed_loop 失敗的 iter 不中斷後續**:可能導致誤導性 summary。**Mitigation**:`status: "failed"` 標記明確,compare_conditions 可選擇 include or exclude
- **decay_factor=0.5 是 magic number**:沒理論基礎。**Mitigation**:CLI flag 可調,實驗週可 ablation 0.1 / 0.5 / 0.9
- **memory db 跨 condition 共用 vs 獨立**:本 change 預設**每個 (exp_name, condition_id) pair 用獨立 db**(`runs/<exp>/<cond>/memory.sqlite`),避免跨 condition 污染。可用 `--memory-db` 覆寫
- **5 iterations × 1500 ep × 25 min ≈ 2 小時 per seed**;6 conditions × 5 seeds = ~60 GPU-hours 完整 run。本 change 不執行,但 `compare_conditions.py` 必須能在數據齊全時跑出結果
- **報告語言**:markdown 報告用英文(與 docstring 一致);圖內字也用英文。中文留給論文章節文字

## Migration Plan

1. 加 scipy 到 pyproject.toml + requirements.txt
2. 改 `train.py::train(config, pre_loaded_buffer=None)`:agent 內部選擇用傳入的還是新建
3. 寫 `hermes_dqn/training/summary.py`:`ClosedLoopSummary` + `IterationSummary` dataclass
4. 寫 `hermes_dqn/training/closed_loop.py`:`run_closed_loop` + CLI `main()`
5. 寫 `tools/compare_conditions.py`:discovery + stats + report
6. Pilot smoke:`python -m hermes_dqn.training.closed_loop --exp-name pilot --condition-id B3-pilot --seed 42 --iterations 3`,觀察 summary.json 與 iter 之間 buffer/memory 是否如預期傳遞
7. Strict validate + commit

Rollback:刪除 closed_loop.py / summary.py / compare_conditions.py + revert train.py 的 optional 參數即可。

## Open Questions

- **B1 hand-shaped reward 誰寫?**`evaluation-criteria` spec 規定「非作者第三人」── 待 demo 前確認
- **decay_factor 最佳值**:0.1 / 0.5 / 0.9 中哪個對 LunarLander 最好,要等 5-seed ablation
- **每 condition 獨立 memory db 還是全 exp 共用?** 本 change 預設獨立(避免 condition 互相干擾);若實驗週發現獨立 db 太空缺乏可用 prior,可改共用
- **iteration 失敗時是否 retry?** 本 change 不 retry(立刻 `status: "failed"` 標記)。實驗週的 batch runner 可包一層 retry 邏輯
- **要不要做「跨 5 seed 平均 fitness 隨 iter 變化」的可視化?** 本 change 的 `iteration_fitness.png` 已涵蓋 per-seed,5-seed-平均的圖可由 compare_conditions.py 補,或者實驗週手動
