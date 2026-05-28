# Part 2 DQN 變體實驗與論文撰寫

> 本段對話在 context 壓縮前進行，以下為摘要重建

## 對話脈絡

### 起因

教授在口試前提問：「只用 vanilla DQN，有沒有考慮 DQN 的其他變體？」
→ 決定加入 Double DQN 與 Dueling DQN（Rainbow 太複雜，選 2/7 個組件）

### 架構決策

**Q：要不要實作 Rainbow DQN？**
A：Rainbow 包含 7 個組件（Noisy Net、PER、Distributional 等），實作量遠超課堂專案規模。
決定選 2 個最具代表性的：
- **Double DQN**：消除最大化偏誤（van Hasselt 2016）
- **Dueling DQN**：V + A 架構分解（Wang 2016）

### 實作重點

**`hermes_dqn/agent/q_network.py`**
- 新增 `DuelingQNetwork` 類別
- Q = V(s) + (A(s,a) − mean_a(A(s,a)))

**`hermes_dqn/agent/dqn_agent.py`**
- `DQNConfig` 加入 `use_double_dqn: bool = False`、`dueling: bool = False`
- `learn()` 分支：Double DQN 用 online net 選動作，target net 評估 Q 值

**CLI 串接**
- `train.py`、`closed_loop.py`、`run_full_experiment.py` 全部加入 `--dqn-variant {vanilla,double,dueling,double_dueling}`

### 實驗設計

```
2 variants × 4 envs × 2 conditions (B0 vs B3) × 5 seeds = 80 runs
runs/part2_double_*/  runs/part2_dueling_*/
```

批次腳本：`scripts/run_overnight_dqn_variants.bat`
（ASCII-only，避免 Windows CMD 的 UTF-8 em-dash 問題）

### 數據分析結論

**核心發現 1：Hermes 在所有 DQN 變體上均顯著優於 B0**
- 所有 4 環境 × 3 變體 pairwise B0 vs B3：p < 0.05

**核心發現 2：不同 DQN 架構間 Hermes 效益無顯著差異**
- Vanilla vs Double vs Dueling 的 Hermes 表現：all pairwise p > 0.3
- 結論：框架具備模型無關性（agent-agnostic）

**密集/稀疏模式**：跨所有三種 DQN 架構一致複現

### 論文 §5 撰寫

結構：
- §5.1 Motivation（為何做 Part 2）
- §5.2 Results — Table 5（B0 vs Hermes 各變體）+ Fig 7
- §5.3 Results — Table 6（Hermes 跨變體穩健性）+ Fig 8

**措辭調整**：
- 初稿：「all pairwise p > 0.3」（min p = 0.3095，太邊緣）
- 修改為：「no pairwise comparison reaches significance (all p > 0.3)」

### GIF 生成

`tools/generate_paper_gifs.py`：4 個環境各生成一個 GIF
- 左：B0-env-native；右：B3-hermes-full
- 3 episodes，lock-step 播放，final frame hold 1 秒
- 輸出：`paper/gifs/{lunarlander,cartpole,mountaincar,acrobot}.gif`

### 圖表生成

`tools/generate_paper_figures.py` 新增：
- `fig7_part2_hermes_vs_b0()`：4 subplot bar chart，含 WIN/n.s. 標註
- `fig8_part2_hermes_robustness()`：Hermes only，跨變體比較

### Reviewer 驗證

`tools/verify_part2.py`：自動審計 Part 2 所有數據
- 確認 5/5 seed、2 conditions、4 envs、2 variants 全部存在
- 確認統計數字與論文一致
