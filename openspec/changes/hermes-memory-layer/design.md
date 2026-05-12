## Context

`gemma-reward-generator` 已完成,證明單發 LLM 寫的 reward 可贏 env-native baseline(seed 42 上 +28% mean、+25pp success、訓練 33% 快)。但目前 Gemma 每次都從零開始 ── 寫完 reward 拿到 fitness 之後,知識就丟掉了。

本 change 加入 README 提到的「四層記憶」中**最重要的第三層 ── SQLite FTS5 長期紀錄**。其餘三層(Short Context / Working Memory / Procedural Memory)在後續 change 補上。理由:長期紀錄是唯一**跨 process / 跨 session / 跨硬碟重啟**仍存活的層,其他三層都是 in-process 的派生物。

硬體與環境限制(由 `env-setup` spec 規定):
- Python 3.11(`sqlite3.sqlite_version` 通常 ≥ 3.40,FTS5 是 SQLite 3.9+ 內建)
- 單一 4090 機器跨 session 共用同一個 `runs/memory.sqlite`
- 不新增任何外部相依(sentence-transformers / embeddings 留給後續若需要)

## Goals / Non-Goals

**Goals:**
- `MemoryStore(db_path)` 介面三呼叫上限:`write(entry)`、`top_k_by_fitness(k, fitness_floor)`、`close()`
- 對既有 baseline 與 gemma-reward-generator 完全 backward-compatible(`--no-memory` 旗標 + 預設 db 缺失自動建立)
- 訓練後內建 100-seed env-native apples-to-apples eval(取代外部 `tools/_eval_env_native.py`),結果寫回記憶
- 對 `LLMRewardClient.generate()` 的擴充不破壞既有 caller(memory 預設空 list)
- Spec scenarios 全部可被 `openspec validate --strict` 通過 + 後續 `closed-loop-fitness` change 可直接引用

**Non-Goals:**
- 其餘三層記憶(Short Context / Working Memory / Procedural Memory)
- 跨任務記憶轉移(LunarLander 紀錄不轉到其他環境)
- Semantic embedding 檢索(只用 FTS5 全文檢索與 SQL ORDER BY)
- 多輪外層迴圈 / 自動迭代閉環(closed-loop-fitness 才做)
- 5-seed 統計比較與 Mann-Whitney U(closed-loop-fitness 才做)
- LLM 自反思(lessons_learned)的 prompt 工程深度調校 ── 第一版用簡單模板就好

## Decisions

### A. 子套件結構

```
hermes_dqn/memory/
├── __init__.py          # re-export MemoryStore / MemoryEntry
├── entry.py             # MemoryEntry dataclass + JSON (de)serialization
├── schema.py            # DDL + migration helpers (single-version for now)
└── store.py             # MemoryStore: connect, write, query, close
```

**理由**:`entry` 是純 dataclass(零相依);`schema` 是 SQL DDL(可單元測試);`store` 是 I/O 包裝。三檔職責分離,後續若要切換 backend(例如改 PostgreSQL)只需替換 `store.py`。

**Alternative considered**:單一 `memory.py` 全部塞在一起。被否決,SQL DDL + Python 邏輯混在一檔可讀性差。

### B. SQLite 表結構(單一 schema,無 migration 機制)

```sql
CREATE TABLE memory (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,             -- ISO 8601 UTC
    run_dir             TEXT    NOT NULL,             -- 對應 runs/<exp>/... 或 runs/<ts>/
    reward_fn_sha256    TEXT    NOT NULL UNIQUE,      -- 防止重複寫入同一份 reward
    reward_code         TEXT    NOT NULL,             -- Python 原始碼字串
    converge_episode    INTEGER,                      -- nullable(可能未收斂)
    mean_reward_last100 REAL    NOT NULL,
    success_rate        REAL    NOT NULL,             -- 0.0–1.0
    env_native_mean     REAL,                         -- nullable(若未跑 apples-to-apples eval)
    env_native_success  REAL,                         -- 同上
    lessons_learned     TEXT,                         -- nullable(可選 LLM 反思)
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_memory_env_native_mean ON memory(env_native_mean DESC);
CREATE INDEX idx_memory_mean_reward     ON memory(mean_reward_last100 DESC);

CREATE VIRTUAL TABLE memory_fts USING fts5(
    reward_code, lessons_learned,
    content='memory', content_rowid='id'
);

-- FTS triggers (insert/delete/update sync)
CREATE TRIGGER memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, reward_code, lessons_learned)
    VALUES (new.id, new.reward_code, new.lessons_learned);
END;
```

**Alternative considered**:
- 雙表(success / failure)分開存:被否決,index 解決排序問題即可,不需 schema 切分
- 把 FTS5 設成 contentless:被否決,觸發 trigger 同步成本可忽略,正常 content table 更直觀

### C. `MemoryStore` API 介面

```python
class MemoryStore:
    def __init__(self, db_path: str | Path):
        """Opens (or creates) the SQLite DB at db_path. Idempotent schema migration."""

    def write(self, entry: MemoryEntry) -> int:
        """Insert entry; returns auto-generated id. UNIQUE(reward_fn_sha256)
        means duplicate writes are silently ignored (returns existing id)."""

    def top_k_by_fitness(
        self,
        k: int = 5,
        fitness_floor: float = 0.0,
        order_by: str = "env_native_mean_or_mean_reward",
    ) -> list[MemoryEntry]:
        """Return the K best entries.

        `order_by` choices:
        - "env_native_mean_or_mean_reward": prefer env_native_mean when not null,
          fall back to mean_reward_last100. This is the recommended default and
          handles legacy entries written before apples-to-apples eval existed.
        - "mean_reward_last100": always use shaped fitness
        - "success_rate": prefer entries with high success_rate
        """

    def all_count(self) -> int:
        """Total entries (for stats / sanity)."""

    def close(self) -> None: ...
```

**理由**:`top_k_by_fitness` 是唯一被 train.py 呼叫的查詢入口。FTS5 暫時不暴露,留給未來「找用了類似 shaping 的 reward」這類查詢。

### D. 防重複寫入(`UNIQUE(reward_fn_sha256)`)

如果同一份 reward 跑了兩次(不同 seed),DB 只保留一筆。新的數據如何處理?

**決議:覆寫**(`INSERT ... ON CONFLICT(reward_fn_sha256) DO UPDATE`),用新的 fitness 蓋過舊的。理由:同 reward 不同 seed 應該被視為「新證據」而非「新 reward」。但這是個 design trade-off,如果未來要做 5-seed 統計,可能要改成多筆並 aggregate。

**Open question**:跨 5 seed 怎麼辦?暫時方案:`reward_fn_sha256` UNIQUE 加上 seed suffix → `<sha>:<seed>`。但這污染了 sha256。Cleaner:加 seed 欄位也納入 UNIQUE constraint。`closed-loop-fitness` 設計 5-seed 比較時再決定。本 change MVP 先用單純 `UNIQUE(reward_fn_sha256)` 覆寫策略。

### E. Prompt 模板擴充

`build_lunarlander_prompt(task_spec, retry_context, force_fallback, prior_attempts=None)`:

當 `prior_attempts` 非空時,在 task_spec 之後 / few-shot 之前插入:

```
PRIOR HIGH-FITNESS ATTEMPTS (use these as inspiration, don't copy):

Attempt A (env_native_mean=207.7, success=0.78):
```python
{reward_code_A}
```
Lessons: {lessons_learned_A or "(none recorded)"}

Attempt B (env_native_mean=183.2, success=0.61):
```python
{reward_code_B}
```
Lessons: ...
```

最多 K 筆,排序方式由 `MemoryStore.top_k_by_fitness` 決定。

**理由**:給 LLM 看「過去什麼有效」是經典 in-context learning。Lessons 段位是非必要(可空),但若存在會幫 LLM 理解 fitness 數字背後的「為什麼」。

### F. train.py 整合流程

```
1. set_global_seed(seed)
2. _make_run_dir()
3. open MemoryStore(memory_db)  [skip if --no-memory]
4. priors = memory.top_k_by_fitness(k=top_k)  [skip if --no-memory]
5. resolve reward:
   - env: stub + None
   - llm: LLMRewardClient.generate(memory=priors)
6. write reward_fn.py + sha256
7. write config.json (incl. memory_state, memory_top_k, memory_priors_used)
8. train DQN (1500 ep)
9. save model_final.pt
10. compute fitness (shaped)
11. compute env_native_mean via inline 100-seed eval
   (port tools/_eval_env_native.py into hermes_dqn/training/eval_env_native.py)
12. (optional) ask Gemma for lessons_learned
13. write MemoryEntry to MemoryStore  [skip if --no-memory or reward_source=env]
14. close MemoryStore
```

**Alternative considered**:把 memory 寫入做成 `02-ending.sh` 階段的離線 task。被否決,訓練後立刻寫入 = 失敗也保留紀錄,跨 process 風險更小。

### G. env 路徑要不要寫入記憶?

**決議:不寫**。env-native baseline 的 reward_code 是固定 stub,寫進去沒有 LLM 學習價值。`--reward-source env` 訓練完畢直接跳過 step 13。但 `--reward-source env` 仍然會跑 step 11 的 env_native_mean(這對 baseline 評估有意義)。

### H. lessons_learned 生成的選擇開關

第一版預設**關閉** lessons_learned 生成(每次訓練只為了寫一句反思就再叫一次 Gemma,API 配額浪費)。透過 `--memory-with-lessons` 旗標啟用。`closed-loop-fitness` 階段可依 budget 決定要不要開。

### I. Reward sandbox L2(子程序隔離驗證)

**動機**:現有 `compile_reward()` 用 `threading.Timer` 限 dry-run 100 ms。但 Python thread 不可被 kill,若 LLM 寫出在 C extension 內無限迴圈(例如 `np.linalg.eig` 餵巨大矩陣)的 reward,thread join timeout 後該 thread 仍佔 CPU 直到程序結束。

**決議**:`compile_reward()` 預設改走子程序驗證流程:

```
parent: 收到 src
parent: spawn multiprocessing.Process(target=_validate_worker, args=(src,))
child:  跑原本的 ast.parse + exec + dry_run(完整流程)
child:  result -> Queue
parent: queue.get(timeout=SANDBOX_TIMEOUT_S=10)
  - 收到 OK -> 主程序 inline re-compile（已驗證安全,re-compile 不會再失敗）
  - 收到 RewardCompileError -> raise(retry loop 走原本邏輯)
  - timeout -> proc.terminate() / proc.kill() -> raise RewardCompileError(stage="subprocess-timeout")
parent: cleanup proc.join()
```

**為什麼通過後要 re-compile 而不傳 callable 回主程序?**因為 callable 不能跨程序傳遞(它持有 closure / globals);傳源碼字串安全,parent 拿到後重做一次 ast.parse + exec(這次不做 dry_run,因為已驗證過)。re-compile 只發生一次/per generate,~10ms,可忽略。

**Memory cap**:`resource.setrlimit(RLIMIT_AS, mem_bytes)` 在 Linux 可,Windows `psutil.Process(pid).memory_info().rss` 監測,超過就 `terminate()`。MVP 先只用 timeout(timeout 內如果配大量記憶體,OS 會 swap 然後 timeout 救援)。

**訓練時的隔離?**訓練每步呼叫 reward_fn 數百萬次,放不進子程序(IPC 1ms × 1500ep × 500steps = 半小時純通訊成本)。**訓練階段 reward_fn 跑在主程序**;我們的論點是:既然 reward 過了子程序驗證(包含 100 ms dry-run + ast 黑名單 + builtins 白名單),它在訓練時造成劫持的機率已經很低 ── 剩下的風險是 logic bug,那是 fitness 數字會告訴我們的事,不是安全問題。

**Alternative considered**:
- **`subprocess` + JSON 通訊**(不用 multiprocessing):被否決,需要重新序列化 import / numpy,還要 worker entry point 腳本檔
- **`signal.SIGALRM`**(Unix-only 真實 timeout):被否決,Windows 無 SIGALRM
- **Docker container**:留給 `reward-sandbox-isolation` change,本 change 不做(setup 1-2 天,對單機課程 final 不對等)

**`--unsafe-inline-compile` 旗標**:debug 用,完全跳過子程序、直接走舊路徑。**不建議生產使用**;只為了讓 LLM 寫的 reward debug 起來更直觀(traceback 行號對得起源碼)。

## Risks / Trade-offs

- **Schema 改變的風險**:本 change 用單一 schema,若後續 change 要加欄位(例如 `seed` 或 `embedding`),需要 migration 機制。**Mitigation**:`schema.py` 提供 `current_version()` 與 `migrate(conn, target_version)`,即使本 change 內 target_version=1,介面留好。
- **SQLite 並行寫入**:本 MVP 假設**單一 process 訓練**(也是現實 ── 4090 一次跑一個訓練)。若未來要平行訓練,需要 WAL 模式 + connection pool。**Mitigation**:現在加 `PRAGMA journal_mode=WAL` 是 cheap insurance,寫進 schema.py。
- **lessons_learned 生成可能寫出長篇大論**:Gemma 反思可能寫很長,污染 FTS5 index。**Mitigation**:prompt 強制要求 ≤ 3 句、用 ast 之類驗證後再寫入。**這個風險暫時不處理**(開關預設關)。
- **跨 5-seed 的覆寫策略**:見 D 的 Open question。本 change MVP 接受「同 reward 不同 seed 會互相覆寫」這個限制,closed-loop-fitness 解決。
- **記憶 schema 漂移**:不同 change 寫入不同欄位的 entry,反序列化失敗風險。**Mitigation**:`MemoryEntry` 所有後續可選欄位都 Optional;`store.py` 讀取時用 `.get()` 不用 `[]`。

## Migration Plan

1. 建 `hermes_dqn/memory/` package + `__init__.py`
2. 實作 `entry.py`(純 dataclass + JSON serializer / deserializer)
3. 實作 `schema.py`(SQL DDL + apply_schema(conn))
4. 實作 `store.py`(`MemoryStore` class)
5. 抽 `tools/_eval_env_native.py` → `hermes_dqn/training/eval_env_native.py::evaluate_on_env_native(model_dir, n=100)`
6. 擴充 `hermes_dqn/llm/prompts.py::build_lunarlander_prompt(..., prior_attempts=None)`
7. 擴充 `hermes_dqn/llm/client.py::LLMRewardClient.generate(..., memory=[])`
8. 改 `hermes_dqn/training/train.py`:加旗標 + 接 MemoryStore
9. Smoke tests:
   - `--no-memory` 路徑 = backward-compatible with current Gemma run
   - `--reward-source llm`(預設記憶開啟)= 空 DB 第一次 run + 第二次 run 看到前一筆記憶
   - `--reward-source env` = 仍然產出 env_native_mean 寫進 config 但不寫入 memory
10. 二次端到端:跑 2 次 `--reward-source llm --seed 42` 與 `--reward-source llm --seed 43`,檢查第二次 prompt 確實含第一次的 reward + fitness 段落

Rollback:刪除 `runs/memory.sqlite` + 用 `--no-memory` 旗標跑即回到 `gemma-reward-generator` 行為。`hermes_dqn/memory/` 目錄與既有 baseline 程式碼正交,刪除即可。

## Open Questions

- **跨 seed 怎麼存** ── MVP 用 UNIQUE(sha256) 覆寫;closed-loop-fitness 階段再決定要不要改成 `(sha256, seed)` 二元 key
- **`top_k_by_fitness` 排序欄位** ── 預設 `env_native_mean_or_mean_reward`,但若 lessons_learned 強烈影響 LLM 表現,排序可能要再加上 lessons 的有無作為次要鍵
- **lessons_learned 的 prompt 工程** ── 用什麼模板問 Gemma 反思?第一版用最簡單的「請給 3 句反思」即可,效果差再調
- **記憶滿了怎麼辦** ── 1500 entry 後 prompt 會撐爆?MVP 假設不超過 100 entries(實驗預算 ≤ 60 GPU-hr,單次 ~25 min,最多 ~144 entries)。若超過,fitness_floor 自然過濾掉低分項
