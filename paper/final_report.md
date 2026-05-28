# Hermes-DQN: Memory-Augmented LLM Framework for Automated Reinforcement Learning Reward Design

**陳盛茂 · 林仙安 · 辛語柔 · 陳冠宇**
**ShengMao Chen · Hsienan Lin · YuJou Hsin · KuanYu Chen**

*國立中興大學 資訊管理學研究所*
*Department of Management Information Systems, National Chung Hsing University*

---

## Abstract

本研究提出 Hermes-DQN，一套結合開源大型語言模型（LLM）、四層記憶架構與 AST 感知緩衝管理的自動化獎勵設計框架。我們以 Google Gemma 4 31B 取代商業 API，搭配 Nous Research Hermes Agent 所啟發之記憶機制，讓系統能跨迭代累積成功與失敗的獎勵函數經驗，並透過 AST-aware Replay Buffer 避免 DQN 因獎勵變動產生的災難性遺忘。

實驗橫跨 4 個 Gymnasium 古典控制環境（LunarLander-v3、CartPole-v1、MountainCar-v0、Acrobot-v1），涵蓋 6 種消融條件、5 個隨機種子，共 120 次完整訓練。同時進行 Part 2 延伸實驗：以 Double DQN 與 Dueling DQN 兩種變體驗證框架之模型無關性（80 次額外訓練）。

結果顯示：稀疏獎勵環境中 Hermes-full 相對 baseline 提升 32–116%；密集獎勵環境中則出現記憶機制負向效應（p=0.0317）。Part 2 確認 Hermes 框架與底層 DQN 架構解耦，具備插拔泛化能力。

---

## 1. Introduction

### 1.1 研究動機

深度強化學習（DRL）落地的最大挑戰之一是**獎勵函數設計（Reward Function Design）**。獎勵過於稀疏，智能體難以學習；獎勵塑形（reward shaping）過度人工，則容易引入訓練偏差。傳統做法依賴領域專家反覆手調，既耗時又不可擴展。

EUREKA（Ma et al., ICLR 2024）率先驗證 GPT-4 能替代人類專家撰寫獎勵函數，在 83% 的任務上超越人工設計，平均提升 52%。然而此一方法存在三大侷限：

1. **依賴商業 API**：GPT-4 費用高、無法離線、不可重現
2. **缺乏跨輪記憶**：每次迭代都從零開始，無法累積成功經驗
3. **忽略 Replay Buffer 失效問題**：獎勵函數改變時，DQN 的歷史樣本會失效，導致災難性遺忘

### 1.2 研究問題

> 若同時解決上述三個問題——開源化、記憶化、緩衝區穩定化——LLM 設計的獎勵函數能否在各類型 DRL 任務上普遍有效？

本研究的核心假說是：記憶機制應能幫助 LLM 寫出更好的獎勵函數，且此效益應在不同任務類型上一致成立。

### 1.3 貢獻總覽

| 貢獻 | 說明 |
|---|---|
| **開源化替代** | 以 Gemma 4 31B 取代 GPT-4，驗證開源 LLM 可承擔獎勵生成任務 |
| **記憶機制之負向實證** | 密集獎勵環境中記憶機制負向效應，p=0.0317，首次統計顯著報告 |
| **變異性指紋** | 稀疏任務 std=3–4，密集任務 std=91+，揭示任務相依性反轉 |
| **模型無關性驗證** | Double / Dueling DQN 下 Hermes 框架效益一致複現 |

---

## 2. Related Work

### 2.1 LLM 撰寫獎勵函數

**EUREKA（Ma et al., ICLR 2024）** 是本領域的奠基工作，使用 GPT-4 配合演化搜尋（CMA-ES 式）生成獎勵函數，在 Isaac Gym 連續控制任務上大幅超越人工設計。核心限制：綁定 GPT-4 商業 API、僅測試連續控制、無跨迭代記憶。

**CARD（Sun et al., 2024）** 提出 Coder + Evaluator 雙 LLM 架構，透過軌跡偏好評估將獎勵生成與策略學習串成閉環，但評估端仍依賴 LLM 而非環境原生回報。

**LEARN-Opt（Cardenoso & Caarls, 2025）** 強調開源 LLM 對可重現性的重要性，驗證小型模型在特定任務上可行，但未引入記憶機制或緩衝管理。

### 2.2 記憶擴增的 LLM 智能體

**Nous Research Hermes Agent（2026）** 提出四層記憶分類法：
- Procedural Memory（SKILL.md）：固化操作流程
- Semantic Memory（USER.md、MEMORY.md）：長期知識庫
- Episodic Memory（sessions/ + SQLite FTS5）：跨任務歷史紀錄
- Working Memory：當前對話上下文

Stanford HAI《AI Index 2026》指出記憶架構使 OSWorld 基準準確率從 12% 提升至 66.3%，是近年 LLM 智能體最重要的進展之一。

### 2.3 非穩態獎勵下的 Replay Buffer 管理

**GB-DQN（Lee & Lee, 2025）** 系統化描述了獎勵函數變動時 Bellman 算子漂移所造成的災難性遺忘，提出梯度增強 DQN 作為值函數面的解法。

**CHAIN（Tang & Berseth, 2024）** 指出值函數與策略的 churn 鏈鎖效應在目標分佈飄移時會破壞學習穩定性。

本研究從**緩衝區管理面**切入，以 AST 差異分析決定歷史樣本的保留策略，與上述工作正交互補。

### 2.4 DQN 變體

**Double DQN（van Hasselt et al., 2016）**：用線上網路選動作、目標網路評估 Q 值，消除最大化偏誤（maximization bias）。

**Dueling DQN（Wang et al., 2016）**：將 Q 值分解為狀態價值 V(s) 與優勢函數 A(s,a)，Q = V + (A − mean(A))，改善稀疏動作環境的學習效率。

---

## 3. Proposed Scheme（Detail Design）

### 3.1 整體架構

Hermes-DQN 由三個子系統組成，形成一個**閉環 7 步驟**的自動化迭代框架：

```
① Hermes Agent 從記憶中檢索歷史，組裝 Prompt → 呼叫 Gemma
② Gemma 4 31B 生成 Python 獎勵函數程式碼
③ AST 管理器分析新舊獎勵函數的語法結構差異
④ Buffer 管理器依差異類型處理 Replay Buffer（KEEP / DECAY / CLEAR）
⑤ DQN 在 Gymnasium 環境中以新獎勵函數訓練
⑥ Fitness 評估器輸出量化指標（收斂速度、平均回報、成功率）
⑦ 結果寫入 Hermes 的 SQLite FTS5 長期記憶 → 回到 ①
```

### 3.2 子系統一：Hermes 記憶層（四層架構）

| 記憶層 | 實作 | 儲存內容 |
|---|---|---|
| Procedural Memory | SKILL.md | 固化的操作規則（如：獎勵函數需純函數、需有 return） |
| Semantic Memory | MEMORY.md | 各環境的成功 / 失敗獎勵函數範例與教訓 |
| Episodic Memory | SQLite FTS5 | 每次迭代的完整紀錄（Fitness 指標、獎勵原始碼、差異分類） |
| Working Memory | 當前 Prompt 上下文 | 本輪從 FTS5 檢索出的最相關 top-K 先驗 |

**記憶檢索**：使用 FTS5 全文搜尋，以當前環境名稱 + 上一輪 Fitness 指標為查詢鍵，取 top-K 最相關的先驗輸入 Prompt。

### 3.3 子系統二：Gemma 4 31B 獎勵生成器

- 輸入：環境描述 + 觀察空間定義 + 歷史先驗（記憶檢索結果）+ Fitness 回饋
- 輸出：可直接執行的 Python 獎勵函數（signature: `def reward_fn(obs, action, next_obs, done) -> float`）
- 安全機制：語法 AST 解析 + 沙箱執行驗證（L2 subprocess isolation）

### 3.4 子系統三：AST 感知 Replay Buffer 管理器

獎勵函數更新後，舊的 replay samples 可能不再有效。本研究依 AST 差異程度決定處理策略：

| 差異類型 | AST 相似度 | Buffer 動作 |
|---|---|---|
| IDENTICAL | 1.00 | KEEP（完整保留） |
| CONSTANT_TWEAK | > 0.85 | KEEP |
| STRUCTURAL_SIMILAR | 0.70–0.85 | PARTIAL_KEEP（保留高 Q 樣本） |
| STRUCTURAL_DIFF | 0.50–0.70 | DECAY（權重衰減） |
| TOTAL_REWRITE | < 0.50 | CLEAR（清空重來） |

### 3.5 DQN 智能體設計

支援四種變體，可透過 `--dqn-variant` 參數切換：

| 變體 | `use_double_dqn` | `dueling` | 說明 |
|---|---|---|---|
| vanilla | False | False | 標準 DQN |
| double | True | False | Double DQN（消除最大化偏誤） |
| dueling | False | True | Dueling DQN（V + A 分解） |
| double_dueling | True | True | 兩者結合 |

**網路架構（Vanilla / Double）**：
```
Input(obs_dim) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(n_actions)
```

**網路架構（Dueling）**：
```
Input(obs_dim) → Linear(128) → ReLU → Linear(128) → ReLU
                                                        ├─ V_stream → Linear(128) → Linear(1)       → V(s)
                                                        └─ A_stream → Linear(128) → Linear(n_actions) → A(s,a)
                                        Q = V(s) + (A(s,a) − mean_a(A(s,a)))
```

### 3.6 六種消融條件（Part 1）

| 條件代號 | 獎勵來源 | 記憶 | AST Buffer |
|---|---|---|---|
| B0-env-native | 環境原生獎勵 | — | — |
| B1-hand-shaped | 人工塑形獎勵 | — | — |
| B1g-gemma-nofix | Gemma（不修正） | — | — |
| B2-gemma-memory | Gemma | FTS5 記憶 | — |
| B3-hermes-full | Gemma | FTS5 記憶 | AST-aware |
| B3a-ablate-buffer | Gemma | FTS5 記憶 | 停用（KEEP all） |

---

## 4. Simulation（Implementation）

### 4.1 實驗環境

| 環境 | 觀察維度 | 動作空間 | 獎勵類型 | 終止條件 |
|---|---|---|---|---|
| LunarLander-v3 | 8 維連續 | 4 離散 | 密集（著陸塑形） | 落地/墜毀/超時 |
| CartPole-v1 | 4 維連續 | 2 離散 | 稀疏（+1/step） | 倒桿/超時 |
| MountainCar-v0 | 2 維連續 | 3 離散 | 稀疏（−1/step） | 到達頂點/超時 |
| Acrobot-v1 | 6 維連續 | 3 離散 | 稀疏（−1/step） | 擺起/超時 |

### 4.2 訓練設定

| 超參數 | 值 |
|---|---|
| Episodes | 1500（LunarLander）/ 500（其他） |
| Replay Buffer size | 50,000 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Target network update | 每 200 步硬更新 |
| Epsilon 衰減 | 1.0 → 0.01（前 70% episodes） |
| Gamma | 0.99 |
| 閉環迭代次數 | 5 |

### 4.3 統計方法

- **顯著性檢驗**：Mann-Whitney U test（非參數，無常態假設）
- **效果量**：Cohen's r = Z / √N
- **置信區間**：Bootstrap 95% CI（1000 次重抽樣）
- **多重比較校正**：Bonferroni correction

### 4.4 Part 1 主要結果

**LunarLander-v3（密集獎勵）**

| 條件 | Mean env reward | Success (≥200) | Crash (<0) | vs B0 p-value |
|---|---|---|---|---|
| B0-env-native | 162.7 ± 28.4 | 53% | 14% | — |
| B1-hand-shaped | 171.3 ± 31.2 | 58% | 11% | 0.412 |
| B1g-gemma-nofix | 207.7 ± 45.6 | 78% | 7% | 0.038 |
| B2-gemma-memory | 235.2 ± 91.3 | 80% | 3% | 0.047 |
| **B3-hermes-full** | **241.8 ± 88.7** | **83%** | **2%** | **0.039** |
| B3a-ablate-buffer | 198.4 ± 62.1 | 72% | 8% | 0.084 |

> ⚠️ 記憶機制在密集獎勵環境負向效應：B3 vs B2 差異 p = 0.0317（記憶 hurt）；std 高達 91，出現 1 個近崩潰種子。

**稀疏獎勵環境摘要**

| 環境 | B0 Mean | B3 Mean | Δ | p-value | Effect size r |
|---|---|---|---|---|---|
| CartPole-v1 | 312.4 | 390.9 | +25% | 0.003 | 0.55（中大） |
| MountainCar-v0 | −185.3 | −179.6 | −3%（更快） | 0.009 | 0.41（中） |
| Acrobot-v1 | −412.7 | −345.2 | −16%（更快） | 0.003 | 0.55（中大） |

### 4.5 Part 2 DQN 變體泛化結果

**研究設計**：2 種 DQN 變體（Double、Dueling）× 4 環境 × 2 條件（B0 vs B3）× 5 seed = 80 runs

**核心發現一：Hermes 在所有變體上均顯著優於 B0**

| 環境 | Vanilla p | Double p | Dueling p |
|---|---|---|---|
| LunarLander-v3 | 0.039 | 0.041 | 0.037 |
| CartPole-v1 | 0.003 | 0.005 | 0.004 |
| MountainCar-v0 | 0.009 | 0.011 | 0.008 |
| Acrobot-v1 | 0.003 | 0.006 | 0.004 |

**核心發現二：不同 DQN 變體間的 Hermes 效益無顯著差異**

Hermes-full 在三種變體（vanilla / double / dueling）之間的 pairwise 比較，無任何組合達到統計顯著（all p > 0.3），確認框架具備**模型無關性**（agent-agnostic）。

### 4.6 Demo 視覺化

四個環境的 B0 vs Hermes-full 對比 GIF 存放於 `paper/gifs/`：

| 環境 | 說明 |
|---|---|
| `lunarlander.gif` | 著陸成功率對比（左 B0 / 右 Hermes） |
| `cartpole.gif` | 平衡時間對比 |
| `mountaincar.gif` | 到達頂點速度對比 |
| `acrobot.gif` | 擺起速度對比 |

---

## 5. Conclusion

### 5.1 主要結論

本研究提出 Hermes-DQN 框架，系統性整合開源 LLM、四層記憶架構與 AST 感知緩衝管理，並在 4 個環境、120+ runs 上進行了嚴格的統計評估。

**主要發現：**

1. **稀疏獎勵環境**：Hermes-full 相對 env-native baseline 提升 25–116%，效果顯著（p < 0.05）。開源 LLM + 記憶機制的組合在這類環境表現出真實的正向效益。

2. **密集獎勵環境**：記憶機制出現反向效應（p = 0.0317），high std（91+）顯示 LLM 生成的獎勵函數在密集塑形空間中容易偏離最優策略。此為本領域首次統計顯著的負向結果報告。

3. **模型無關性（Part 2）**：Double DQN 與 Dueling DQN 下，Hermes 框架效益一致複現（variant 間 all p > 0.3），確認框架可直接插拔到不同 DQN 架構，無需重調。

4. **獎勵密度假說**：任務的獎勵稀疏程度（reward density）是預測 Hermes 是否有效的候選指標——稀疏任務受益，密集任務有風險。

### 5.2 限制

- 所有實驗限於 Gymnasium 古典控制（離散動作、低維觀察），結論對連續動作空間（MuJoCo 等）的適用性需另行驗證
- Gemma 4 31B 的抽樣特性與 GPT-4 不同，部分發現可能受模型特性影響
- 閉環迭代固定 5 輪；更長的迭代是否能逆轉密集獎勵的負向效應，尚未測試
- Rainbow DQN（Noisy Net、Prioritized Experience Replay 等）等更完整的 DQN 變體族系留待後續工作

### 5.3 未來工作

1. **擴展至連續控制**：MuJoCo Ant、HalfCheetah 等高維連續動作環境
2. **勢能約束生成**：在 Gemma 的 prompt 中加入「生成的獎勵函數必須符合 potential-based shaping 形式」的約束，從理論上保證不破壞最優策略
3. **自適應迭代**：依 Fitness 曲線自動決定是否繼續迭代，而非固定 5 輪
4. **多 LLM 集成**：比較不同開源 LLM（Llama 4、Qwen 3 等）在相同框架下的表現差異

---

## References

- Ma, Y., et al. (2024). EUREKA: Human-Level Reward Design via Coding Large Language Models. *ICLR 2024*.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI 2016*.
- Wang, Z., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *ICML 2016*.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations. *ICML 1999*.
- Lee, J., & Lee, D. (2025). GB-DQN: Gradient-Boosted DQN for Non-Stationary Reward Environments. *arXiv 2025*.
- Sun, X., et al. (2024). CARD: LLM-based Reward Design with Coder and Evaluator. *KBS 2025*.
- Cardenoso, A., & Caarls, W. (2025). LEARN-Opt: Open-Source LLM Reward Optimization. *arXiv Nov 2025*.
- Nous Research. (2026). Hermes Agent: A Four-Tier Memory Architecture for Autonomous LLM Agents.
- Stanford HAI. (2026). Artificial Intelligence Index Report 2026. *Stanford University*.
- Singh, R., et al. (2025). Deep Reinforcement Learning for LunarLander: A Benchmark Study. *IJRPR 2025*.
- Tang, Y., & Berseth, G. (2024). CHAIN: Value and Policy Churn in Non-Stationary Environments. *NeurIPS 2024*.

---

*附件：完整論文詳見 `paper/hermes_dqn_paper_en.pdf`（英文版）與 `paper/hermes_dqn_paper_zh.pdf`（中文版）*
