## Why

`gemma-reward-generator` 證明了單次 Gemma 寫的 reward 可以贏過 env-native(+28% mean,+25pp success),但每次都是「**第一次見面的代課教練**」── 完全不記得自己上次寫了什麼、效果如何。

這份 change 接上 README 三大核心貢獻的第 2 點「**記憶擴增**」── 給 Gemma 一本不會忘記的筆記本,讓它跨輪次站在前一份 reward 的肩膀上改進。對應到文獻:OSWorld 加上記憶後性能從 12% → 66.3%,記憶對 LLM agent 是已知有巨大效應的設計。

本 change 完成後,`experiments-protocol` spec 規定的 6 個 condition 中,有 3 個變成可比較:
- B2 EUREKA-style(Gemma 一發、無記憶)
- B3 Hermes-DQN full(這份 change + 後續)
- B3-no-memory(關掉記憶但保留其他)

「**記憶讓 fitness 提升了 X**」這個論文要 claim 的數字才有來源。

## What Changes

- 新增 `hermes_dqn/memory/` 子套件:
  - `entry.py`:`MemoryEntry` dataclass(id / timestamp / run_dir / reward_fn_sha256 / reward_code / converge_episode / mean_reward_last100 / success_rate / env_native_mean / lessons_learned)
  - `schema.py`:SQLite 表 + FTS5 virtual table 的 DDL 與 migration
  - `store.py`:`MemoryStore(db_path)` 提供 `write(entry)` / `top_k_by_fitness(k=5, fitness_floor=0)` API
- 擴充 `hermes_dqn/llm/`:
  - `client.py::LLMRewardClient.generate()` 接受 `memory: list[MemoryEntry] = []`
  - `prompts.py::build_lunarlander_prompt()` 新增 `prior_attempts` 參數,prompt 加入 "PRIOR HIGH-FITNESS ATTEMPTS" 區塊
- `train.py` 修改:
  - 新增 `--memory-db <path>`(預設 `runs/memory.sqlite`)
  - 新增 `--memory-top-k <K>`(預設 5)
  - 新增 `--no-memory` 旗標(關閉記憶讀寫,供 ablation 用)
  - 訓練前:從 DB 讀 top-K → 餵進 `LLMRewardClient`
  - 訓練後:跑內建 apples-to-apples eval(100 unseen seeds)算 `env_native_mean`
  - 訓練後:把該次結果寫回 DB
- (可選)`lessons_learned` 反思:訓練後再呼叫 Gemma 一次,問它「這份 reward 為什麼好/壞」,1-3 句存進 `MemoryEntry`
- 把 `tools/_eval_env_native.py` 的核心邏輯抽進 `hermes_dqn/training/eval_env_native.py` 讓 train.py 直接呼叫(不再走子程序)
- `config.json` 新增欄位:`memory_state`(`"none"` 或 `"hermes-sqlite-fts5"`)、`memory_top_k`、`memory_priors_used`(餵了哪些 entry id)
- **L2 reward sandbox**:`hermes_dqn/llm/sandbox.py` 新增 `validate_reward_in_subprocess(src, timeout_s, memory_mb)`,在獨立 `multiprocessing.Process` 跑完整 compile + dry-run;失敗(timeout / OOM / 例外)由父程序 `terminate()` + `kill()` 確保隔離;通過後父程序 inline re-compile 取得 callable 給訓練用(無 IPC 開銷)。`compile_reward()` 預設改走此路徑,可用 `--unsafe-inline-compile` 旗標關閉(僅供 debug)

## Capabilities

### New Capabilities

- `memory-store`:SQLite FTS5-backed 長期記憶儲存,封裝 schema 管理、條目寫入、top-K 查詢
- `memory-llm-integration`:把記憶條目編組為 prompt 上下文,並在訓練後自動把新結果寫回
- `reward-sandbox`:LLM 寫的 reward 程式碼在 `multiprocessing.Process` 內驗證,timeout / OOM 由父程序 hard-kill;通過後再回主程序 inline 編譯(訓練時零 IPC 開銷)── Docker-like 隔離的 L2 子程序版本,Windows + Linux 通用,L3 容器化留給後續 `reward-sandbox-isolation` change

### Modified Capabilities

- `llm-reward-client`:
  - `generate()` 新增 `memory` 參數;若提供 ≥ 1 條,prompt 加入 "PRIOR HIGH-FITNESS ATTEMPTS" 區塊
  - `compile_reward()` 強化沙箱 ── 由 thread-based timeout(目前的弱保護)改為 `multiprocessing.Process` 子程序驗證 + hard kill;通過後主程序 inline re-compile 拿 callable

（train.py 整合 `MemoryStore` 讀寫的部分由新 capability `memory-llm-integration` 全權處理 ──`llm-reward-integration` 的現有 Requirement 仍成立,只是 train.py 同時也滿足新的 memory 相關 Requirement,沒有衝突需要 MODIFY。）

## Impact

- 新增檔案:`hermes_dqn/memory/{__init__,entry,schema,store}.py`、`hermes_dqn/training/eval_env_native.py`、`hermes_dqn/llm/sandbox.py`
- 修改檔案:`hermes_dqn/llm/client.py`、`hermes_dqn/llm/prompts.py`、`hermes_dqn/llm/compile.py`(改走子程序驗證)、`hermes_dqn/llm/__init__.py`、`hermes_dqn/training/train.py`、`hermes_dqn/__init__.py`
- 新增相依:**無**(SQLite + FTS5 是 Python 3.11 stdlib 內建;`multiprocessing` 是 stdlib;Windows + Linux 通用)
- 新增 artifact:`runs/memory.sqlite`(gitignored,跨 run 累積;若刪除即重啟「無記憶」狀態)
- 不破壞既有 run:`--no-memory` 旗標 + 預設行為(若 db 不存在會自動建)讓 `bootstrap-dqn-baseline` 與 `gemma-reward-generator` 的 smoke / 1500-ep run 全部仍可重現
- 引用的 spec scenarios:
  - `establish-project-lifecycle-spec / doc-standards`:Requirement "OpenSpec 四件套強制規則" + "程式碼註解 WHY-only 原則"
  - `establish-project-lifecycle-spec / experiments-protocol`:Requirement "Condition Triple Definition"(memory_state 從 `none` 改 `hermes-sqlite-fts5` 屬於新 condition)
  - `establish-project-lifecycle-spec / env-setup`:Requirement "Dependency Lockfile Committed"(本 change 不新增相依,僅需確認 sqlite3 stdlib 即可)
  - `bootstrap-dqn-baseline / fitness-evaluation`:Requirement "FitnessReport data shape" 與 "FitnessEvaluator reads JSONL logs"(`MemoryEntry` 內嵌 fitness)
  - `gemma-reward-generator / llm-reward-client`:Requirement "Generate Python reward source code"(擴充既有 generate() 簽章不破壞)
