# 治理 Spec(`establish-project-lifecycle-spec`)— 多 Agent 協作紀錄

**OpenSpec change**:`establish-project-lifecycle-spec`
**模式**:**3 個討論 subagent + 1 個 reviewer subagent**(共 4 個 agent 平行 / 串行)
**產出**:5 個 capability spec,**46 個 Requirement**,strict validate 一次通過

---

## 觸發點

使用者問:
> 那如果拆分成很多步驟 從文件規範 環境設置 實際實驗 評估結果 文件完成
> 請你派三個subagent討論其中相關的細節 並在派一個subagent做review 做成一個這個專案的spec
> 記住 要能部屬在本地端4090 且預期可收斂的DQN

→ 5 個生命週期階段 + 3 個討論者 + 1 個 reviewer = 4 個 agent 的協作架構

---

## Agent 分工

| Agent | 角色 | 涵蓋階段 | 字數限制 |
|---|---|---|---|
| 1 | 文件 lead | 文件規範 + 文件完成 | ≤ 1500 |
| 2 | 基礎建設 lead | 環境設置 | ≤ 1500 |
| 3 | 實驗 / 評估 lead | 實際實驗 + 評估結果 | ≤ 1800 |
| 4(Reviewer) | 整合 + 寫 OpenSpec change | 全部 | 自由 |

### 為什麼 3 個討論者不是 5 個

5 階段中:
- 「文件規範」(寫文件的規矩)和「文件完成」(交什麼文件)是同個人腦袋的事
- 「實際實驗」和「評估結果」綁在一起(實驗設計決定能算什麼統計)
- 「環境設置」自己一塊

→ 自然 grouping = 3。

### Agent prompt 設計重點

每個 prompt 都包含:
1. **共同 context**(專案介紹、4090 / Python 3.11 / DQN 收斂目標、4 人團隊、已完成 baseline)
2. **必讀檔案清單**(CLAUDE.md、README、白話介紹、相關 spec)
3. **要做的決策清單**(每個都列具體選項,要 agent 「取一個立場」而非列 pro/con)
4. **輸出格式範本**(Decisions / Draft Requirements / Risks / Open Questions 同一模板,reviewer 才能拼接)
5. **約束**(不寫檔,只回文字;具體數字勝過 generic 描述)

---

## Agent 提案摘要

### Agent 1(文件)

亮點:
- OpenSpec design.md 是「條件性必要」── 引入新外部相依、新跨子系統介面、或 capability ≥ 2 時才寫
- 註解 **英文 only, WHY-only**
- 第 N 版升版規則:change archive、教師審閱、重大實驗 → 升;純錯字 → 不升
- 論文 IEEE 雙欄 8 章 8-12 頁;4 人各包 1-2 章

### Agent 2(環境)

亮點:
- **uv 為主**(比 pip 快 10-100×)、pip + venv 備援
- Python 3.11.x 鎖定(3.12 改了 ast 模組會影響 ast-buffer-manager)
- CUDA `cu121` 鎖定不升級(4090 已驗證,升級無收益)
- pre-commit:`ruff check --fix` + `ruff format` + `openspec validate --strict`(< 3 秒)
- 跳過 mypy/pytest 在 pre-commit(太慢)
- 4090 預約:`docs/4090-booking.md` + `scripts/claim-gpu.sh` 用 `nvidia-smi` 守門

### Agent 3(實驗 + 評估)

亮點:
- **5 seeds per condition**,seeds `[42, 43, 44, 45, 46]`
- Condition = `(reward_fn_source, memory_state, buffer_policy)` 三元組
- **固定 1500 ep,禁 early-stop**(公平的 converge_episode 比較)
- Hermes 外層 5 次 LLM 重寫
- **6 baseline**:B0 env-native / B1 hand-shaped / B2 EUREKA-style / B3 full / B3-no-memory / B3-no-AST
- 統計檢定:**Mann-Whitney U + 5000 bootstrap CI**
- Win = `p<0.05 AND ≥10% 平均差 AND CI 不重疊`(三條件齊全)
- Primary metric:**converge_episode**(secondary: success_rate)
- Outlier 全部 report 不剔除
- **可重現七件套**:git SHA / 5 seed / reward_fn.py / config.json / episodes.jsonl / pyproject.toml / Gemma prompt 日誌(脫敏)

---

## Reviewer 解決的衝突

| 衝突點 | Reviewer 的決議 | 理由 |
|---|---|---|
| 環境管理工具(uv vs pip+venv) | **uv 主 + pip 備援**;`uv export` 自動產 requirements.txt | 速度收益大,但保留 fallback 不阻斷任何隊員 |
| run/ 目錄(flat vs hierarchical) | **混合制**:dev/ad-hoc 用 flat timestamp;eval 級強制 `runs/<exp>/<condition>/seed_NN/` | 既有 baseline 不破壞,新格式給統計用 |
| 註解語言(中文 vs 英文) | **英文 only** | identifier 是英文,混排不一致 |
| 4090 排隊(Issues vs nvidia-smi) | **雙層**:Issues 預約 + nvidia-smi 守門 | 人會忘 Issues,腳本兜底 |
| Seed 數(5 vs 8) | **5 起步,pilot 顯示 effect 小再升 8** | n=5 + Mann-Whitney 平衡 |

---

## 產出檔案結構

```
openspec/changes/establish-project-lifecycle-spec/
├── proposal.md      # Why / What Changes / Capabilities / Impact
├── design.md        # 5 階段的決策 + 衝突收斂 + Risks + Migration + Open Questions
├── tasks.md         # 3 group(Author / Validate / Post-merge adoption)
└── specs/
    ├── doc-standards/spec.md           # 9 Requirements
    ├── env-setup/spec.md               # 11 Requirements
    ├── experiments-protocol/spec.md    # 8 Requirements
    ├── evaluation-criteria/spec.md     # 10 Requirements
    └── final-deliverables/spec.md      # 8 Requirements
```

**46 個 Requirement,每個都有 ≥ 1 個 testable Scenario(4-hashtag heading, WHEN/THEN 格式)。**

---

## 這份 spec 怎麼影響後續所有 change

任何後續 feature change(`gemma-reward-generator` 起算)在 `proposal.md` 必須引用本 spec 對應的 scenario,例如:

> 引用的 spec scenarios(因為 establish-project-lifecycle-spec 已建立治理規範,本 change MUST 引用):
> - establish-project-lifecycle-spec / env-setup:Requirement "API Key Management via .env" 全部 scenarios
> - establish-project-lifecycle-spec / experiments-protocol:Requirement "Reward Function Artifact and Integrity"(reward_fn.py + SHA-256)

`gemma-reward-generator` 已實踐這個引用模式。

---

## 經驗回顧

### 這個 multi-agent 模式何時有用

- ✅ 跨領域的大型決策(本次:文件 / 環境 / 實驗 / 評估 / 交付 5 個面向)
- ✅ 需要多個立場 brainstorm 再選一個(避免單一 agent 偏見)
- ✅ 產出可結構化的 contract(spec.md 模板)
- ❌ 程式碼實作(不適合 — agents 各寫各的會撞)
- ❌ 簡單問答(雞用牛刀)

### 我犯的小錯

- 第一輪 prompt 對 agent 3 給了 1800 字限制,結果他寫到 1800 字邊緣;reviewer agent 要把這份消化進 design.md。下次給 1200 即可
- Agent 2 把 uv 提得很堅持,但 reviewer 還是讓 pip+venv 留為 fallback ── 表示我給 reviewer 的 prompt 鼓勵了「中庸」,符合 4 人團隊現實
