# Hermes 記憶層 + L2 子程序 Sandbox

**OpenSpec change**:`hermes-memory-layer`(已 archive 為 `2026-05-12-hermes-memory-layer`)
**狀態**:全 task 完成、strict-valid、n=1 驗證

---

## 設計

對應 README 三大核心貢獻第 2 點(記憶擴增)的**第一層** ── SQLite FTS5 長期記憶。其餘 3 層(Short Context / Working Memory / Procedural Memory)留給後續 change。

### 4 個 capability spec

| Capability | 內容 | Scenarios |
|---|---|---|
| `memory-store`(新) | `MemoryStore(db_path).write(entry)` / `top_k_by_fitness(k=5)`,SQLite FTS5 + WAL + UNIQUE(sha256) upsert | 5 Req, 11 Scen |
| `memory-llm-integration`(新) | train.py 整合:`--memory-db` / `--memory-top-k` / `--no-memory` 旗標、訓練前讀 priors、訓練後寫回 | 6 Req, 12 Scen |
| `reward-sandbox`(新) | L2 子程序 sandbox:`multiprocessing.Process` + hard-kill timeout + Linux RLIMIT 記憶體上限 | 5 Req, 11 Scen |
| `llm-reward-client`(MODIFIED) | `generate()` 新增 `memory` 參數,prompt 加 "PRIOR HIGH-FITNESS ATTEMPTS" 區塊 | 1 MODIFIED Req |

### 為什麼把 L2 sandbox 塞進這份 change

使用者要求「**hermes 可以限制住,類似 docker 的方式**」── 對話中討論了三層隔離:

- L1:現有 AST 黑名單 + builtins 白名單(`gemma-reward-generator` 提供)
- L2:子程序 + hard-kill(本 change 加)── 主程序不會被 LLM 程式碼拖死
- L3:Docker 容器(獨立 `reward-sandbox-isolation` change,proposal-only)

L2 子程序的關鍵設計:**驗證跑在子程序、訓練時 reward_fn 跑回主程序內**。理由:訓練每 step 都呼叫 reward_fn,放子程序內每步 IPC 會慢 1000×。子程序只負責「驗證」這一次性步驟。

### 重要設計決策

| 決策 | 理由 |
|---|---|
| **SQLite FTS5 + WAL** | 內建 stdlib,無新外部相依;WAL 跨 process 安全 |
| **UNIQUE(reward_fn_sha256) + ON CONFLICT 覆寫** | 同 reward 不同 seed 視為新證據;5-seed 統計留給 closed-loop-fitness |
| **Top-K 排序用 `COALESCE(env_native_mean, mean_reward_last100)`** | 優先 apples-to-apples,fallback shaped;處理 legacy entries |
| **env 路徑不寫 memory(但仍跑 env-native eval)** | env stub 對 LLM 學習無價值 |
| **lessons_learned 預設關閉** | 省 Gemma API 配額,留給 closed-loop-fitness 決定 |
| **`multiprocessing` spawn 模式** | Windows + Linux 通用,Linux 加 `RLIMIT_AS` 記憶體限 |

---

## 實作過程中的 bug

### Bug 1:np.float32 不通過 `isinstance(result, (int, float))`

LLM 寫的 reward 返回 `np.float32`(因為 `next_obs[4]` 是 float32)。`isinstance(np.float32(0.5), float)` 在 numpy 2.x 上 False。

**Fix**:改用 `try: float(dry_result)` 嘗試轉換。

### Bug 2:fitness_floor=0.0 過濾了 undertrained entries

`top_k_by_fitness(k=5)` 預設 `fitness_floor=0.0`(spec 規定的「production 級門檻」)。但 smoke run 的 env_native_mean=-553 完全被過濾。

**Fix**:train.py 呼叫時加 `fitness_floor=float("-inf")` 覆寫,讓早期 entry 也能當 prior。Spec 預設值不變(那是給 mature production 用)。

### Bug 3:LLM 寫的 reward 把舊 buffer 的 weights 變不確定

(這個是後續 ast-buffer-manager 才發現的)

### Bug 4:Windows + 多進程的 RLIMIT 不可用

`resource.setrlimit` 是 POSIX-only。Windows 上沒有。

**Fix**:加 `if sys.platform.startswith("linux")` 條件,Linux 上設 RLIMIT,Windows 上純靠外部 timeout 守門(降級但仍能用)。

---

## Smoke + 驗證結果(全綠)

### Sandbox 5/5 unit case(`tools/_smoke_sandbox.py`)

```
PASS  Valid reward source (validated in 0.78s)
PASS  Syntax error -> 'syntax-error'
PASS  import os -> 'ast-import-rejected'
PASS  while True: pass -> 'dry-run-timeout' (caught by inner cap, also valid)
PASS  Wrong arity -> 'signature-arity'
```

### Training 5/5 case

```
8.1 env path deterministic vs baseline ── byte-identical 前 10 集
8.3 First memory run (空 DB) ── entry id=1 寫入
8.4 Second memory run ── 讀 1 prior,prompt 含 PRIOR HIGH-FITNESS ATTEMPTS
8.5 env-native eval inline ── 每 run config.json 有 env_native_mean
8.6 --no-memory 跳過寫入 ── DB row count 不變
```

### 1500-ep × 2 (`runs/gemma_mem_seed42` + `gemma_mem_seed43`)

| run | priors | env_native_mean | success | crash |
|---|---|---|---|---|
| `gemma_mem_seed42`(空 memory) | `[]` | **235.21** | 80% | **3%** |
| `gemma_mem_seed43`(讀 seed 42) | `[1]` | 224.53 | 78% | **3%** |

對照 `gemma_seed42`(無 memory,207.72 / 78% / 7%):**memory 機制端到端跑通、crash rate 砍半**。
但 n=1 看不出 memory 是否「讓 Gemma 寫更好的 reward」── 留給統計實驗。

---

## 產出檔案

```
hermes_dqn/memory/
├── __init__.py
├── entry.py       # MemoryEntry dataclass(11 欄)
├── schema.py      # SQLite DDL + FTS5 + WAL + version
└── store.py       # MemoryStore CRUD + 上下文管理器

hermes_dqn/llm/
└── sandbox.py     # validate_reward_in_subprocess + 子程序 worker

hermes_dqn/training/
└── eval_env_native.py  # 從 tools/_eval_env_native.py 抽進 package
```

修改:`hermes_dqn/llm/{compile,prompts,client,__init__}.py`、`hermes_dqn/training/train.py`(memory 整合 + inline eval)。

---

## 多 agent 協作回顧

這份 change 本身**沒用**多 agent ── 設計是線性的(一個人想完即實作)。但**前一份治理 spec** 是 3+1 agent 模式,本份 change 引用了那些 spec 的 scenario(env-setup R5、experiments-protocol R5),確認多 agent 協作的設計**真的被後續 change 引用到**。
