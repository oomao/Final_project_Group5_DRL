## Context

`hermes-memory-layer` 提供了 Hermes 教練的長期筆記(SQLite FTS5)。但**球員端的「肌肉記憶」(replay buffer)目前不在記憶範圍內** ── 每次訓練都是空 buffer 從零開始。

`closed-loop-fitness` 將實作 5 次 LLM iteration 的外層迴圈。一個常見的 question:第 2 次迭代開始時,要不要把第 1 次累積的 buffer 沿用?直覺答案是「看新舊 reward 差多少」── 這正是 README 提到的 AST 感知緩衝區的核心。

本 change 把這個判斷與處理流程拆成**純函式庫**:
- AST 差異分類器
- Buffer 動作 enum + 決策函式
- Replay buffer 的持久化 + 衰減機制

`closed-loop-fitness` 在多輪迭代之間呼叫這些函式;本 change 自身**完全不接 train.py**。這個切分讓:
1. AST diff classifier 可以獨立 unit test(synthetic 程式碼對比)
2. Buffer save/load 可以獨立驗證(寫盤、讀盤、確定性)
3. `closed-loop-fitness` 的 scope 縮小到「主迴圈 + 統計」

## Goals / Non-Goals

**Goals:**
- `diff_rewards(old_src, new_src)` 對 4 種典型差異模式輸出正確分類
- `decide_policy(diff)` 確定性對應到 `KEEP / DECAY / CLEAR`
- `ReplayBuffer.save(path)` / `load(path)` 跨 process 還原 buffer 狀態 byte-identical
- `ReplayBuffer.decay_weights(factor)` 配合 weighted sampling,讓 `KEEP + decay` 真的讓舊樣本被抽到的機率降低
- 既有 `ReplayBuffer.sample()` 在 weights 全 1.0 時 byte-identical to before
- 全部 unit test 可在 < 2 秒內跑完(純 numpy + ast,不需 GPU 也不需訓練)

**Non-Goals:**
- 多輪迭代訓練主迴圈(`closed-loop-fitness`)
- 跑大量實驗驗證「DECAY 真的比 CLEAR 好」(也是 `closed-loop-fitness` 統計階段的事)
- PER(Prioritized Experience Replay 用 TD error)── DECAY 用簡單 age-weight 即可
- 跨任務 buffer 轉移(LunarLander → CartPole)
- 比 4 類更細的 diff 分類(IDENTICAL / NUMERIC / STRUCTURAL / TOTAL 已足夠)

## Decisions

### A. 子套件結構

```
hermes_dqn/buffer/
├── __init__.py           # re-export RewardDiff / BufferAction / 主要函式
├── ast_diff.py           # diff_rewards(old_src, new_src) -> RewardDiff
├── policy.py             # BufferAction enum + decide_policy(diff)
└── rebuild.py            # apply_policy(buffer, action, decay_factor)
```

**理由**:跟 `hermes_dqn/memory/` 一樣的「pure function + side-effecting class」切分。AST diff 是純函式(無 I/O),buffer 處理有副作用(寫 buffer state)。

### B. RewardDiff 分類規則

```python
@dataclass(frozen=True)
class RewardDiff:
    kind: Literal["IDENTICAL", "NUMERIC_DIFF", "STRUCTURAL_DIFF", "TOTAL_REWRITE"]
    similarity: float  # [0.0, 1.0], 1.0 for IDENTICAL
    diff_summary: str  # human-readable, e.g. "3 constants changed: 0.1->0.2, 0.5->0.3, 100->150"
```

分類流程:
1. 若 `sha256(old) == sha256(new)` → `IDENTICAL`(similarity=1.0)
2. 將兩份程式碼用 `ast.parse` 取 AST
3. 抽出「結構簽章」── `[node_type, name_or_const_placeholder, ...]` 的序列
4. 若兩個簽章序列完全相等 → `NUMERIC_DIFF`(similarity=1.0,但 bytes 不同)
5. 否則用 `difflib.SequenceMatcher(None, sig_a, sig_b).ratio()` 算相似度
6. similarity > 0.7 → `STRUCTURAL_DIFF`,反之 → `TOTAL_REWRITE`

**結構簽章具體規則**:
- 對每個 AST node 收 `type(node).__name__`
- 若是 `ast.Name` 加 `.id`
- 若是 `ast.Attribute` 加 `.attr`
- 若是 `ast.Constant` 且值為數字 → 加 `"<NUM>"` placeholder(隱藏數值,只看結構)
- 若是 `ast.Constant` 其他類型(string / bool / None)→ 加 `repr(value)`

**Alternative considered**:
- 用 `astor.dump_tree(remove_ws=True)` 比較 ── 被否決,多個外部相依;`ast.walk` 夠用
- 用 tree-edit-distance(Zhang-Shasha 演算法)── 被否決,實作複雜,對我們的場景 SequenceMatcher 已足夠

### C. BufferAction 對應表

| Diff | Action | 原理 |
|---|---|---|
| `IDENTICAL` | `KEEP` | 完全相同,buffer 任何處理都沒意義 |
| `NUMERIC_DIFF` | `KEEP` | 只調整係數,Q-value 估計仍大致有效 |
| `STRUCTURAL_DIFF` | `DECAY` | 結構有變但相似,舊經驗仍有部分價值,降低權重 |
| `TOTAL_REWRITE` | `CLEAR` | 完全不同的 reward,舊樣本可能反向誤導 |

`decide_policy` 是純函式 + lookup table,無 side effect。

### D. ReplayBuffer 擴充:weighted sampling

新增欄位:
```python
self._weights = np.ones((capacity,), dtype=np.float32)
```

`sample(batch_size)`:
```python
weights = self._weights[: self._size]
probs = weights / weights.sum()
idx = self._rng.choice(self._size, size=batch_size, p=probs, replace=True)
# Return same Batch tuple as before
```

`push(...)`:新樣本的 weight 自動設為 1.0(覆蓋舊位置時也是)。

`decay_weights(factor=0.5)`:`self._weights[:self._size] *= factor`(只衰減已存在的樣本)。

**Backward-compat**:當 `_weights` 全為 1.0,`probs` 是 uniform,`np.random.choice(p=uniform)` 與 `np.random.integers` 雖然 seed 相同但**生成序列不同**(內部演算法不同)── 這會破壞 baseline 的 byte-determinism!

**Mitigation**:在 `sample` 開頭判斷:
```python
if np.all(self._weights[: self._size] == 1.0):
    # uniform fast-path; identical to legacy code
    idx = self._rng.integers(0, self._size, size=batch_size)
else:
    weights = self._weights[: self._size]
    probs = weights / weights.sum()
    idx = self._rng.choice(self._size, size=batch_size, p=probs, replace=True)
```

這條 fast-path 確保:既有 baseline / gemma / memory 跑(從未呼叫 `decay_weights`)的 sample 序列 byte-identical。

### E. Persistence schema

```python
def save(self, path):
    np.savez_compressed(
        path,
        obs=self._obs[: self._size],
        actions=self._actions[: self._size],
        rewards=self._rewards[: self._size],
        next_obs=self._next_obs[: self._size],
        dones=self._dones[: self._size],
        weights=self._weights[: self._size],
        idx=np.int64(self._idx),
        size=np.int64(self._size),
        capacity=np.int64(self.capacity),
        rng_state=np.array([self._rng.bit_generator.state], dtype=object),
    )
```

**為何只存 `[:_size]`?**:capacity 100K,但若實際只填了 5K 條經驗,存空欄位是浪費(~7.6 MB → ~400 KB 壓縮後)。

**Load**:還原 capacity → 重建 zero 陣列 → 把 `[:size]` 填回 → 還原 `_idx`、`_size`、`_rng.bit_generator.state`。

**Alternative considered**:用 pickle ── 被否決,跨 Python 版本不穩定。`np.savez` 是 numpy 內建,二進位格式穩定。

### F. apply_policy 的副作用界定

```python
def apply_policy(
    buffer: ReplayBuffer,
    action: BufferAction,
    decay_factor: float = 0.5,
) -> None:
    """Mutates buffer in place. No-op if action is KEEP."""
    if action is BufferAction.KEEP:
        return
    if action is BufferAction.DECAY:
        buffer.decay_weights(decay_factor)
        return
    if action is BufferAction.CLEAR:
        buffer.clear()  # new method
        return
    raise ValueError(f"Unknown BufferAction: {action!r}")
```

`buffer.clear()` 新增方法:把 `_obs / _actions / _rewards / _next_obs / _dones / _weights` 全部歸零,`_idx = _size = 0`。**不重置 `_rng`**(rng state 跟著 buffer 走比較合理,但這是 design 取捨)。

### G. AST diff 對「不可解析的程式碼」的處理

理論上 ast_diff 只被叫來比較已通過 sandbox 驗證的 reward 程式碼(語法保證合法)。但為了健壯:
- 若 `ast.parse(old_src)` raise `SyntaxError`,則直接 fallback 到 `TOTAL_REWRITE`(老的不能解析,當作完全不同)
- 同理 new_src

不 raise 例外,避免單一壞輸入卡住整個迭代。

## Risks / Trade-offs

- **Sampling fast-path 維護成本**:每次 sample 多一個 `np.all(weights == 1.0)` 檢查(O(n))。**Mitigation**:n ≤ 100K,檢查耗時 < 0.1 ms,可忽略
- **decay_factor 沒理論基礎**:0.5 是直覺值,可能不對。`closed-loop-fitness` 階段可以 ablation 0.1 / 0.5 / 0.9 看哪個效果好
- **similarity threshold 0.7 是 magic number**:類似上面,屬於可在後續實驗微調的常數;本 change 寫死先求簡單
- **AST parse 在某些 edge cases 可能誤判**(例如 LLM 把運算式重排但語意相同 → AST 結構不同 → STRUCTURAL_DIFF 即使語意 IDENTICAL)。**Mitigation**:文件說明這是「保守估計」,寧可錯判為 DECAY 也不要錯判為 KEEP
- **Buffer save/load 的 RNG state 還原**:numpy bit_generator state 保證跨 process 一致,但若未來換 RNG(PCG64 → Philox)需要重新驗證

## Migration Plan

1. 建 `hermes_dqn/buffer/` package
2. 實作 `ast_diff.py`(`ast.walk` + `difflib.SequenceMatcher`)
3. 實作 `policy.py`(enum + lookup)
4. 擴充 `ReplayBuffer`(weights + sample fast-path + save/load + decay_weights + clear)
5. 實作 `rebuild.py::apply_policy`
6. Smoke / unit tests:
   - AST diff: 5 synthetic 程式碼對比(IDENTICAL / NUMERIC / STRUCTURAL / TOTAL / unparseable)
   - Buffer save/load: 寫 10K transitions → save → 新 instance load → 確認 array 全等 + 下次 sample 與原 buffer 一致
   - Decay: 推 5K transitions → decay(0.5) → 推 5K transitions → sample 1000 次 → 統計舊樣本佔比 ≈ 0.5/(0.5 + 1.0) = 1/3
   - Backward-compat: 不呼叫 decay 時,sample 序列 byte-identical to baseline ReplayBuffer

Rollback:刪除 `hermes_dqn/buffer/` + revert `replay_buffer.py` 的 3 個新方法即可。

## Open Questions

- `decay_factor` 的最佳值 ── 等 `closed-loop-fitness` 5-seed 比較才知道
- similarity threshold 0.7 是否該分階段(例如 NUMERIC vs STRUCTURAL 之間還有 "minor structural" 介於 0.85-1.0)── 暫定簡單 4 類,看後續實驗結果再說
- 是否需要 `buffer.merge(other_buffer)` ── 跨任務轉移時可能需要,但本 change scope 之外
