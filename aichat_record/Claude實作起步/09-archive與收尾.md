# Archive 7 個 Change + Session 收尾

**狀態**:7 個 strict-valid change 全部 archive、19 個永久 capability spec、`main` 推到 GitHub。

---

## 歸檔順序與依賴

OpenSpec archive 會把 `openspec/changes/<name>/specs/` 合併到 `openspec/specs/<capability>/spec.md`。如果 change 之間有 MODIFY 依賴,順序要先 archive base 才能 archive modifier(否則 modifier 無法找到 target spec)。

| # | Change | 創造的 spec | 修改的 spec | 依賴 |
|---|---|---|---|---|
| 1 | `improve-dev-scripts` | dev-startup, dev-ending | — | — |
| 2 | `bootstrap-dqn-baseline` | dqn-baseline, reward-plugin, fitness-evaluation | — | — |
| 3 | `establish-project-lifecycle-spec` | 5 governance(doc-standards / env-setup / experiments-protocol / evaluation-criteria / final-deliverables) | — | — |
| 4 | `gemma-reward-generator` | llm-reward-client, llm-reward-integration | — | (bootstrap reward-plugin) |
| 5 | `hermes-memory-layer` | memory-store, memory-llm-integration, reward-sandbox | **llm-reward-client** | (gemma) |
| 6 | `ast-buffer-manager` | ast-buffer-manager | **dqn-baseline** | (bootstrap) |
| 7 | `closed-loop-fitness` | closed-loop-engine, condition-comparison | — | (上面全部) |

按這個順序跑 `openspec archive <name> --yes`,7 個全成功。

---

## archive 中發現的 bug:ast-buffer-manager 的 spec 結構錯了

第一次 archive `ast-buffer-manager` 時被拒:

```
ast-buffer-manager: target spec does not exist; only ADDED requirements
are allowed for new specs. MODIFIED and RENAMED operations require an
existing spec.
```

原因:我把 ADDED Requirements(for 新 capability `ast-buffer-manager`)和 MODIFIED Requirements(for 既有 capability `dqn-baseline`)塞在**同一個檔** `specs/ast-buffer-manager/spec.md`。

正確結構:
- `specs/ast-buffer-manager/spec.md` → 只放新 capability 的 ADDED
- `specs/dqn-baseline/spec.md` → 只放對既有 capability 的 MODIFIED + 新增的 ADDED

切成兩檔之後 archive 成功。這條經驗(連同 OpenSpec 對 delta spec 檔的命名/結構規則)寫進該次 commit 訊息,給後續任何**同時新增 capability + 修改既有 capability** 的 change 用。

---

## 最終 OpenSpec 全景

### `openspec/specs/`(19 個永久 capability)

```
ast-buffer-manager     dev-ending           dev-startup
doc-standards          env-setup            experiments-protocol
evaluation-criteria    final-deliverables   dqn-baseline
fitness-evaluation     reward-plugin        llm-reward-client
llm-reward-integration memory-llm-integration  memory-store
reward-sandbox         closed-loop-engine   condition-comparison
numbering-rule
```

### `openspec/changes/`

- `archive/2026-05-12-*/`:7 個已 archive 的 change
- `reward-sandbox-isolation/`:proposal-only,觸發條件式(等課程教師要求 / Hugging Face 公開 / 多人共用 4090 才實作)

---

## 本 session 的 commit 全景

```
8ec758d  Implement Hermes-DQN baseline, governance spec, and Gemma reward generator
ee53098  Update README: add progress table, empirical results, quick-start, file tree
55c560b  Add Hermes memory layer + L2 reward sandbox; propose L3 container isolation
fde4948  Add ast-buffer-manager: AST-aware reward diff + replay buffer policy library
7e9e9d4  Add closed-loop-fitness: multi-iter Hermes-DQN loop + statistical comparison
2d2f5a2  Run closed-loop-fitness pilot: 3 iter x seed 42, mechanism verified
2526bb2  Archive 7 strict-valid changes; consolidate into 19 permanent capabilities
```

7 個 commit、~8000 行新程式碼、Author 全部 `oomao <csm088220@gmail.com>`、無 Claude attribution。

---

## 三大核心貢獻的最終實作狀態

| README 三大貢獻 | 對應 capability | 驗證證據 |
|---|---|---|
| **開源化**(Gemma 取代 GPT-4) | `llm-reward-client` + `llm-reward-integration` + `reward-sandbox` | smoke 7/7;Gemma 1500-ep 完勝 baseline(207.72 vs 162.72,+28%) |
| **記憶擴增**(SQLite FTS5 長期層) | `memory-store` + `memory-llm-integration` | smoke 5/5;閉環 pilot 驗證跨 iter 傳遞 priors |
| **AST 感知緩衝區** | `ast-buffer-manager` + `dqn-baseline` MODIFIED | unit 27/27;pilot 觀察到正確的 DECAY / CLEAR 套用 |

加碼(README 未明列但確實做了):

- **L2 子程序 sandbox**(防 LLM 程式碼把訓練程序拖死)
- **7 步閉環引擎**(`closed-loop-engine`)
- **Mann-Whitney + bootstrap 統計工具**(`condition-comparison`)
- **治理 spec 46 條**(governance 5 個 capability)

---

## Session 之後留給人手做的

| 項目 | 卡點 |
|---|---|
| **B1 hand-shaped reward** 撰寫 | 治理 spec 規定要**非作者第三人**寫,需找組員 |
| **完整 6 cond × 5 seed × 5 iter 實驗** | ~60 GPU-hr,4090 排程 |
| **論文 §4-5 撰寫** | 等實驗數據 |
| **Demo 影片 v2** | 手動錄製 |
| **`reward-sandbox-isolation` L3 容器化** | 觸發條件式,可選 |

不是「設計風險」── 是「工時 + 人手」。

---

## 知識傳承:這 9 份對話紀錄涵蓋什麼

| # | 主題 |
|---|---|
| 01 | Session 總覽:起點、終點、決策摘要 |
| 02 | DQN baseline 實作(`bootstrap-dqn-baseline`)|
| 03 | 白話架構文件設計(籃球教練比喻)|
| 04 | 治理 spec 多 agent 協作(3+1 模式)|
| 05 | Gemma reward 接入(`gemma-reward-generator`)|
| 06 | 記憶層 + L2 sandbox(`hermes-memory-layer`)|
| 07 | AST 緩衝區(`ast-buffer-manager`)|
| 08 | 7 步閉環 + 統計工具(`closed-loop-fitness`)|
| **09** | **archive + 收尾(本檔)** |

下個接手者:
1. `npm run dev:start` 自動讀 `04-handover.md` 看 Next Actions
2. 想懂專案脈絡?讀 `白話架構介紹.md` + 這 9 份對話紀錄
3. 想寫新 change?讀 `openspec/specs/<capability>/spec.md` 然後 `/opsx:propose`
