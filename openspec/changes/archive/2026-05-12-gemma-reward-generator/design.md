## Context

`bootstrap-dqn-baseline` 已完成,vanilla DQN 在 LunarLander-v3 收斂並可重現。`establish-project-lifecycle-spec` 已定義專案治理規範,本 change 必須引用其 env-setup 與 experiments-protocol 的對應 scenario。

目標環境:Windows 11 + Python 3.11 + RTX 4090 + cu121,與既有環境共用 venv。Gemma 4 31B 透過 Google AI Studio API 取用;沒有 API key 時,`--reward-source llm` 路徑必須以清楚錯誤訊息退出,不得影響 `--reward-source env`(baseline)路徑。

LLM 生成 Python 程式碼後 `exec()` 為 callable 是這份 change 的**最大風險面**:Gemma 可能寫出語法錯誤、import 錯誤、執行時 raise、或回傳非 float。design 必須把這個風險面收斂到一個明確的 boundary。

## Goals / Non-Goals

**Goals:**
- `LLMRewardClient(api_key, model)` 一行可建立;`.generate(task_spec)` 回傳 Python 原始碼字串
- `compile_reward(src)` 把字串轉成符合 `RewardFunction` Protocol(7 args)的 callable;失敗自動重試 ≤ 3 次,每次重試把錯誤訊息 + traceback 喂回 Gemma 要它修
- `train.py --reward-source llm` 端到端可跑 1500 ep,產出 `episodes.jsonl` + `reward_fn.py` + `config.json` 含 SHA-256
- 不破壞 `--reward-source env` 的 baseline 路徑
- 沒有 `GOOGLE_API_KEY` 時 `--reward-source llm` 立即退出,訊息明確指向 `.env.example`

**Non-Goals:**
- Hermes 四層記憶(`hermes-memory-layer` 負責)
- AST 差異分析(`ast-buffer-manager` 負責)
- 多輪 LLM iteration(`closed-loop-fitness` 負責,本 change 只跑單一 LLM 生成)
- 跨 5 seed 的統計比較(只證一次能收斂,multi-seed evaluation 另立 change)
- prompt 自動搜尋(prompt 是固定模板,效果搜尋留給後續)
- 細部 reward function 安全沙箱(只做 ast 黑名單 + `__builtins__` 限制,不做 seccomp/Docker)

## Decisions

### 子套件結構

```
hermes_dqn/llm/
├── __init__.py          # re-export LLMRewardClient, compile_reward
├── client.py            # LLMRewardClient(api_key, model="gemma-3-27b-it")
├── compile.py           # compile_reward(src) + RewardCompileError
└── prompts.py           # build_lunarlander_prompt() + few-shot 範例
```

**理由**:`client` 處理外部 API I/O、`compile` 處理字串→callable 的純邏輯、`prompts` 把可調整的提示集中(後續做 prompt 比較時不用動其他檔)。

**Alternative considered**:把全部放單一 `llm.py`。被否決,因為 prompt 模板會變動,單一檔太擠且難 review。

### LLM 模型字串

預設 `model="gemma-3-27b-it"`(Google AI Studio 目前可叫的 Gemma 開源版),`.env` 可用 `GEMMA_MODEL=` 覆寫。README 說的「Gemma 4 31B」是未來目標,實作時優先確認 Google AI Studio 當下提供的 Gemma 模型字串,以 `.env` 覆寫即可不改程式碼。

**Alternative considered**:hardcode `gemma-4-31b`。被否決,API 上模型名稱會變動,讓 `.env` 蓋過更穩。

### 7-arg RewardFunction Protocol 相容性

Gemma 必須寫出符合既有 Protocol 的函式:

```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info) -> float
```

**prompt 設計**:in-context 教 LLM:函式簽章是哪 7 個 args、obs 是 `np.ndarray shape (8,)`、action 是 `int ∈ {0,1,2,3}`、回傳 `float`。附 1-2 個 few-shot 範例(passthrough + 簡單油門懲罰)。

**Alternative considered**:讓 LLM 寫 class 或 lambda。被否決,top-level `def reward(...)` 解析最簡單,也容易 SHA-256 hash。

### Compile + exec 沙箱

`compile_reward(src)` 流程:
1. `ast.parse(src)` ── 拒絕 import 任何模組(允許清單只有空,因為函式只能用 obs/action/next_obs/info dict,info 可能含 numpy)
2. `exec(compiled, restricted_globals)` ── `restricted_globals` 只暴露 `{'np': numpy, '__builtins__': SAFE_BUILTINS}`;`SAFE_BUILTINS` 是白名單,允許 `abs/min/max/sum/len/range/float/int/bool/dict/list/tuple/print` 等
3. 從 namespace 取 `reward` 名稱,檢查是 callable + 簽章參數數 == 7
4. 用一筆模擬 transition(隨機 obs/action/next_obs)測 dry-run 一次,確認回傳是 float、不 raise

**失敗處理**:任一步失敗 raise `RewardCompileError(stage, message, traceback)`,由 caller 決定要不要重試。

**Alternative considered**:
- 直接 exec 不限制 → 被否決,LLM 可能寫 `import os; os.system(...)` 之類
- Docker 沙箱 → 被否決,過於工程化,單機開發無法承受
- restricted_python → 被否決,額外相依、過時專案

**Risks accepted**:有經驗的攻擊者可能繞過 ast 黑名單(例如 `getattr(__builtins__, '__import__')`)。**這份 change 不防範對抗式輸入**;LLM 並非對抗者,只是會犯錯,**WHY-only 註解一律記:`# sandbox: trust model not to be adversarial, only to make mistakes`**。

### LLMRewardClient.generate() retry 邏輯

```
attempt 1: 純 prompt (含任務描述 + protocol + few-shot)
   → compile_reward → 成功就回傳
attempt 2: prompt + 上次錯誤 + 「請修正以下問題,維持原本 def reward(...)」
   → compile_reward → 成功就回傳
attempt 3: prompt + 上次錯誤 + 強制要求「使用最簡單的 fallback:直接 return env_reward」
   → compile_reward → 成功就回傳
attempt 4 失敗:raise RewardGenerationError,train.py 退出非 0
```

**理由**:三次重試在 EUREKA 觀察中已足以收斂在「至少能編譯的版本」。第 3 次強制 fallback 確保最壞情況不會把整個訓練流程鎖死。

**Alternative considered**:無限重試。被否決,Google AI Studio 配額會被吃光。

### `train.py` 整合介面

新增旗標:

```bash
python -m hermes_dqn.training.train --reward-source llm --episodes 1500 --seed 42
```

新增環境變數(從 `.env` 載入):
- `GOOGLE_API_KEY`(必需,當 `--reward-source llm` 時)
- `GEMMA_MODEL`(可選,預設 `gemma-3-27b-it`)

`train()` 函式內加分支:
```python
if reward_source == "llm":
    src = LLMRewardClient(api_key).generate(LUNARLANDER_TASK)
    reward_fn = compile_reward(src)  # 含 retry
    (run_dir / "reward_fn.py").write_text(src, encoding="utf-8")
    config_data["reward_fn_sha256"] = sha256(src.encode()).hexdigest()
else:
    reward_fn = None  # baseline path
env = make_env(seed=config.seed, reward_fn=reward_fn)
```

**reward_fn.py 永久化**:**所有** run(不只 LLM 路徑)都應寫入 `reward_fn.py`。env-source 路徑寫一行註解 `# env native reward (no custom function)` 的 stub,SHA-256 用 stub 內容算。**理由**:統一 artifact 結構,evaluation 腳本不用 if/else。

### `.env.example`

```
# Required when --reward-source llm
GOOGLE_API_KEY=

# Optional override (default: gemma-3-27b-it)
GEMMA_MODEL=
```

提交到 git;`.env` 自己仍在 `.gitignore` 內。

## Risks / Trade-offs

- **Gemma 寫出無效程式碼率高**:第一次跑可能要 3 次重試才過。**Mitigation**:設 timeout 60s/次 + 完整 traceback 喂回 prompt;若 3 次後仍失敗,記錄 prompt + 4 次回應到 `runs/<ts>/llm_attempts.jsonl` 方便日後 prompt 工程
- **Google AI Studio 配額**:免費 tier 15 RPM。**Mitigation**:本 change 一次 run 只呼叫一次 LLM(≤ 3 次重試),不會撞配額;`closed-loop-fitness` 才會撞,屆時加配額管理
- **API key 洩漏**:**Mitigation**:`.env` 在 `.gitignore`,`.env.example` 值為空,pre-commit `AIza` regex 偵測(env-setup R6 已規)
- **編譯沙箱被繞過**:見上面「Risks accepted」。LLM 是合作者而非攻擊者,黑名單足夠
- **Reward 函式跑太慢拖累訓練**:Gemma 可能寫複雜算式。**Mitigation**:dry-run 階段 timeout 100ms/次,超過拒絕
- **訓練收斂變差或不收斂**:這份 change 不保證 LLM reward 「比 baseline 好」,只保證「能跑通管線」;`closed-loop-fitness` 才負責比較統計

## Migration Plan

1. 加 google-genai / python-dotenv 到 `pyproject.toml` + `requirements.txt`
2. 寫 `hermes_dqn/llm/{prompts, compile, client}.py`(由內到外:純邏輯 → API 包裝)
3. 改 `train.py` 接 `--reward-source` 旗標 + reward_fn.py 永久化
4. 新增 `.env.example`,叮嚀 README 加一段 Gemma key 申請流程
5. **smoke**:`--reward-source llm --episodes 10 --seed 42` 跑通即視 Gate A 過
6. **end-to-end**:`--reward-source llm --episodes 1500 --seed 42` 收斂(不要求贏 baseline,只要求 `total_episodes == 1500`、`reward_fn.py` 存在、SHA-256 一致)
7. 寫 `## Baseline + Gemma 一次性對照` 一行到 `hermes_dqn/README.md`

Rollback:`--reward-source env` 路徑與 baseline 一致;只要使用者不傳 `--reward-source llm` 旗標就完全回到 baseline 行為。移除 `hermes_dqn/llm/` 目錄即可完全撤回。

## Open Questions

- Google AI Studio 當下提供哪些 Gemma 模型字串?Apply 階段第一步要先 `genai.list_models()` 確認,並把結果寫進 hermes_dqn/README.md 的 troubleshooting 段
- Prompt 用中文還英文?**先英文**(LLM 寫程式邏輯時英文 prompt 普遍較穩),範例與註解可中文。Apply 過程若英文版產出品質不佳,改中文 prompt 並補上 design 註記
- Temperature 設多少?**先 0.7**(留一些創造力),`closed-loop-fitness` 階段可改 0 做可重現
- 要不要把整段 Gemma 回應(包括非程式碼的解釋)存起來?**要**,存到 `runs/<ts>/llm_attempts.jsonl`,每行一次 attempt;便於 prompt 後驗
