# AST 感知緩衝區管理器

**OpenSpec change**:`ast-buffer-manager`(已 archive 為 `2026-05-12-ast-buffer-manager`)
**狀態**:全 task 完成、strict-valid、27/27 unit case 通過

---

## 設計

對應 README 三大核心貢獻第 3 點 ── 換 reward 時用 AST 判斷該保留 / 衰減 / 清空 replay buffer,避開災難性遺忘(Bellman 漂移、Churn Chain 效應)。

### 切分:純函式庫 vs 訓練整合

本 change 只做「**判斷器 + 處理器**」純函式;訓練主迴圈的整合留給 `closed-loop-fitness`。

理由:
1. AST diff 純函式,可獨立 unit test(synthetic 程式碼對比)
2. Buffer save/load 可獨立驗證
3. 縮小 `closed-loop-fitness` 的測試難度

### 2 個 spec 檔(踩到 archive 才發現要切分)

| Spec 檔 | 內容 |
|---|---|
| `specs/ast-buffer-manager/spec.md` | 新 capability,3 個 ADDED Requirement(RewardDiff 分類、BufferAction enum、apply_policy) |
| `specs/dqn-baseline/spec.md` | 對既有 dqn-baseline capability:2 MODIFIED(Vanilla DQN arch 加 fast-path、Reproducible training 加 save/load RNG)+ 1 ADDED(ReplayBuffer persistence and reset) |

第一次 archive 時把 MODIFIED 跟 ADDED 全塞同一檔,被 archiver 拒絕:
> ast-buffer-manager: target spec does not exist; only ADDED requirements are allowed for new specs.

切成兩檔之後 archive 成功。這條經驗寫進 commit 訊息,給未來「同時新增 capability + 修改既有 capability」的 change 參考。

### 4 種 RewardDiff 分類

| Kind | 觸發條件 | BufferAction |
|---|---|---|
| `IDENTICAL` | bytes 一致 | KEEP |
| `NUMERIC_DIFF` | AST 結構相同,僅數字字面值改變 | KEEP |
| `STRUCTURAL_DIFF` | AST 不同但相似度 > 0.7 | **DECAY** |
| `TOTAL_REWRITE` | 相似度 ≤ 0.7 | **CLEAR** |

相似度演算法:抽出 AST 節點型別 + Name + Attribute 序列(忽略數字常數值),用 `difflib.SequenceMatcher.ratio()`。

### ReplayBuffer 擴充

| 新方法 | 用途 |
|---|---|
| `save(path)` | `np.savez_compressed` 全部 array + idx + size + RNG state |
| `load(path)` | 反序列化還原,RNG state 還原讓 sample 序列 byte-identical |
| `decay_weights(factor)` | 對 `_weights[:_size]` 乘以 factor(只衰減已存在樣本) |
| `clear()` | 全部歸零、idx/size = 0、不重置 RNG |

新增 `_weights: np.float32` 欄位。`sample()` 加 fast-path:
```python
if np.all(self._weights[:self._size] == 1.0):
    # legacy uniform path — byte-deterministic with baseline
    idx = self._rng.integers(0, size, size=batch_size)
else:
    probs = weights / weights.sum()
    idx = self._rng.choice(size, p=probs, replace=True)
```

**為什麼需要 fast-path**:`np.random.Generator.integers` 與 `choice(p=uniform)` 即使 seed 相同也產生不同序列。fast-path 確保**從未呼叫 decay_weights 的 caller**(baseline / gemma 等所有既有 run)取樣序列 byte-identical。

---

## Unit test(27/27 通過)

`tools/_smoke_ast_buffer.py` 涵蓋:

| 區塊 | 案例數 | 結果 |
|---|---|---|
| `diff_rewards` | 5(IDENTICAL/NUMERIC/STRUCTURAL/TOTAL/unparseable) | 5/5 |
| `decide_policy` | 4(4 種 diff 各對應 action) | 4/4 |
| `apply_policy` | 4(KEEP/DECAY/CLEAR/unknown raises) | 4/4 |
| ReplayBuffer 向後相容 | 5(weights 預設 1.0、fast-path、decay 套用、新樣本仍 1.0、~1/3 取樣機率) | 5/5 |
| save/load 往返 | 9(6 個 array + 3 個 scalar + RNG state) | 9/9 |

亮點:**取樣機率 1/3 驗證實際觀察值 = 0.327**(對 expected 0.333 偏差 < 2%)。決定論 fast-path 在不呼叫 decay 時與 baseline byte-identical(透過 env 路徑 10 ep run 驗證)。

---

## 產出檔案

```
hermes_dqn/buffer/
├── __init__.py
├── ast_diff.py     # diff_rewards + RewardDiff
├── policy.py       # BufferAction enum + decide_policy
└── rebuild.py      # apply_policy

hermes_dqn/agent/replay_buffer.py(擴充):
├── _weights 欄位
├── sample() 加 fast-path
├── decay_weights() / clear() / save() / load()

tools/_smoke_ast_buffer.py:27 case unit smoke
```

---

## 重要的回顧:何時切 MODIFY 何時切 ADDED

| 場景 | spec 檔結構 | 範例 |
|---|---|---|
| 純新 capability | 1 個檔 `specs/<new-name>/spec.md`,只 `## ADDED Requirements` | `memory-store` |
| 純修改既有 capability | 1 個檔 `specs/<existing-name>/spec.md`,只 `## MODIFIED Requirements` | `llm-reward-client`(被 hermes-memory-layer 修改) |
| **同時新增 + 修改既有** | **2 個檔**:新檔放 ADDED、既有檔放 MODIFIED(可選 ADDED) | **本 change**:`ast-buffer-manager/spec.md` 新增 + `dqn-baseline/spec.md` 修改 |

這個切分規則寫進 commit 訊息,給後續任何同時引入新 capability 並修改既有 capability 的 change 用。
