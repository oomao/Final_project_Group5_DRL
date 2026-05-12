## Why

README 三大核心貢獻的第 3 點是「AST 感知緩衝區」── 當 LLM 換寫 reward 函數時,用靜態程式碼分析判斷新舊 reward 的差異類型,進而決定 replay buffer 該**保留 / 衰減 / 清空**,避開 DQN 在獎勵函數變動下的災難性遺忘(Bellman 運算子漂移 / Churn Chain)。

`hermes-memory-layer` 已讓 Gemma 寫的 reward 可累積跨輪次經驗。但目前每次訓練都是「**空的 replay buffer 從頭開始**」── 如果改成多輪迭代閉環(`closed-loop-fitness` 規劃),自然會問:**前一輪累積的 buffer 該怎麼處理?**

- 沿用會帶舊 reward 標籤的經驗,可能誤導新 reward 下的 Q-value 估計
- 直接清空又浪費了 800-1500 episodes 的探索成本

本 change 提供「判斷器 + 處理器」**作為庫**,純函式可獨立 unit test。實際接進訓練主迴圈由 `closed-loop-fitness` 完成。

## What Changes

- 新增 `hermes_dqn/buffer/` 子套件:
  - `ast_diff.py`:`diff_rewards(old_src, new_src) -> RewardDiff` 將兩份 reward 程式碼分類為 4 種差異:
    - `IDENTICAL`:bytes 一致(SHA-256 相同)
    - `NUMERIC_DIFF`:AST 結構一致,僅數字字面值改變(例如 `0.1 → 0.2`)
    - `STRUCTURAL_DIFF`:AST 結構不同但相似度 > 0.7
    - `TOTAL_REWRITE`:相似度 ≤ 0.7
  - `policy.py`:`BufferAction` enum(`KEEP / DECAY / CLEAR`) + `decide_policy(diff)` 對應規則
  - `rebuild.py`:`apply_policy(buffer, action, decay_factor=0.5)` 對 replay buffer 套用決策
- 擴充 `hermes_dqn/agent/replay_buffer.py`:
  - `save(path)`:`np.savez` 序列化全部 array + idx + size + RNG state 到 `.npz`
  - `load(path)`:反序列化還原 buffer 狀態
  - 新增 `_weights` 欄位(per-sample 取樣權重,預設 1.0)
  - `decay_weights(factor)`:把目前 `_size` 中**全部現存**樣本的權重乘以 `factor`,新樣本仍以 1.0 加入
  - `sample()` 改用 `np.random.Generator.choice(p=normalize(weights))` 帶 weights
  - 維持向後相容:不呼叫 `decay_weights` 時,所有 weights 仍為 1.0,sample 行為與既有 buffer 等價

## Capabilities

### New Capabilities

- `ast-buffer-manager`:reward 程式碼 AST 差異分類 + buffer 處理策略 + buffer 持久化的純函式庫,提供給 `closed-loop-fitness` 與其他後續整合 change 使用

### Modified Capabilities

- `dqn-baseline`:`ReplayBuffer` 既有 `push` / `sample` / `__len__` 介面不破壞;新增 `save` / `load` / `decay_weights`;`sample` 在 weights 全為 1.0 時行為等價於既有版本(deterministic backward-compat)

## Impact

- 新增檔案:`hermes_dqn/buffer/{__init__,ast_diff,policy,rebuild}.py`、`hermes_dqn/buffer/README.md`(可選)
- 修改檔案:`hermes_dqn/agent/replay_buffer.py`(新增 3 個方法 + 1 個欄位)、`hermes_dqn/agent/__init__.py`(re-export)、`hermes_dqn/__init__.py`(可選 re-export)
- 新增相依:**無**(`ast` / `difflib` / `numpy` 都已用)
- **不修改 train.py**:本 change 是純庫,訓練腳本由 `closed-loop-fitness` 處理整合
- 不破壞既有 baseline 與 gemma 訓練:全部 weights 預設為 1.0,sample 行為 byte-deterministic 等價;`save`/`load`/`decay_weights` 是新增方法,既有 caller 不呼叫即無感
- 引用的 spec:
  - `establish-project-lifecycle-spec / doc-standards`:Requirement "OpenSpec 四件套強制規則" + "程式碼註解 WHY-only 原則"
  - `bootstrap-dqn-baseline / dqn-baseline`:Requirement "Vanilla DQN architecture" 與 "Reproducible training"(weighted sampling 的擴充必須維持 backward-compat 確定性)
  - `gemma-reward-generator / llm-reward-integration`:Requirement "Mandatory reward_fn.py artifact per run"(`ast_diff.py` 讀的就是這份 artifact)
