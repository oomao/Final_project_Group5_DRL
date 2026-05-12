## Why

`bootstrap-dqn-baseline` 已證明 DQN 在 LunarLander-v3 上可用環境內建 reward 收斂(seed 42 第 399 ep 收斂、最後 100 ep 平均 262.79、成功率 95%)。Hermes-DQN 的核心命題是「**開源 LLM 寫的 reward function 能讓 DQN 學得更好或更快**」。要驗證這個命題,第一塊磚是把 Gemma 4 31B 接進來,讓它**真的**寫出可執行的 reward 程式碼餵給 DQN。

這份 change 不做記憶、不做 AST、不做多輪閉環 ── 只證明「Gemma 寫的 reward 能跑通端到端訓練並產出 fitness 報告」。對應到 EUREKA 命題(GPT-4 寫 reward)的**開源重現**:把貴的 GPT-4 換成免費的 Gemma。後續所有實驗(記憶 / AST / 閉環)都建立在這條打通的管線上。

## What Changes

- 新增 `hermes_dqn/llm/` 子套件:
  - `client.py`:`LLMRewardClient` 包裝 google-genai SDK,輸入任務描述 → 輸出 Python reward function 原始碼字串
  - `compile.py`:把 LLM 字串 compile + exec 成符合 `RewardFunction` Protocol 的 callable,失敗自動重試最多 3 次
  - `prompts.py`:LunarLander 任務的 prompt 模板(obs/action space 說明、期望輸出格式、few-shot 範例)
- `train.py` 新增 `--reward-source {env,llm}` 旗標:`env`(預設)維持 baseline 行為;`llm` 先呼叫 Gemma 生成 reward,再用它訓練
- run 目錄新增兩個必要 artifact:`reward_fn.py`(實際使用的源碼) + `config.json` 新增 `reward_fn_sha256` 欄位(滿足 `experiments-protocol` R5)
- API key 管理:`.env` 載入 `GOOGLE_API_KEY`,新增 `.env.example` 範本(滿足 `env-setup` R5)
- `requirements.txt` / `pyproject.toml` 加入 `google-genai`、`python-dotenv`

## Capabilities

### New Capabilities
- `llm-reward-client`:封裝 Gemma API 呼叫、prompt 組裝、字串到 callable 的 compile 流程,並處理編譯/執行失敗的自動重試
- `llm-reward-integration`:訓練腳本與 LLM-生成 reward 的整合介面(`--reward-source` 旗標、reward_fn.py 永久化、SHA-256 完整性)

### Modified Capabilities
- none(`dqn-baseline`、`reward-plugin`、`fitness-evaluation` 介面不變;只新增使用方式)

## Impact

- 新增檔案:`hermes_dqn/llm/{__init__,client,compile,prompts}.py`、`.env.example`
- 新增相依:`google-genai`(必需)、`python-dotenv`(必需)
- 修改檔案:`hermes_dqn/training/train.py`(新增 `--reward-source` 旗標 + reward_fn 永久化)、`pyproject.toml` + `requirements.txt`(新增相依)、`hermes_dqn/__init__.py`(re-export 新類別)
- 不破壞 baseline:`--reward-source env`(預設)行為與 baseline 完全一致,既有 `runs/baseline_seed42/` 完整可比較
- 外部依賴:需使用者在 Google AI Studio 申請 API key 並放入 `.env` 才能跑 `--reward-source llm`
- 引用 spec(governance hook):
  - `establish-project-lifecycle-spec / env-setup`:Requirement "API Key Management via .env"(三個 scenarios 全部)
  - `establish-project-lifecycle-spec / experiments-protocol`:Requirement "Reward Function Artifact and Integrity"(reward_fn.py + SHA-256)
  - `bootstrap-dqn-baseline / reward-plugin`:Requirement "Injectable reward in env wrapper"(7-arg 簽章相容)
