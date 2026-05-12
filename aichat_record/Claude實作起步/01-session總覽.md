# Claude 實作起步 — Session 總覽

**日期**:2026-05-12
**AI**:Claude Opus 4.7 (1M context)
**人員**:csm088220@gmail.com (1 名,本 session 為個人推進)
**Session 範圍**:從零實作 Hermes-DQN 的前三塊磚 — DQN baseline、專案治理 spec、Gemma reward 接入

---

## Session 起點 vs 終點

| 面向 | Session 開始 | Session 結束 |
|---|---|---|
| **Python 程式碼** | 完全沒有 | 完整 `hermes_dqn/` package(5 個子模組,~1200 行) |
| **DQN baseline** | 設計只在 README | 訓練可跑通,seed=42 收斂於 ep 399、success 95% |
| **治理規範** | 散落 / 未成文 | 46 個 SHALL/MUST Requirement,5 個 capability spec,strict validate 通過 |
| **LLM reward 通道** | 概念 | 實作完成,Gemma 第一次寫就過編譯,smoke 7/7 全綠 |
| **OpenSpec changes** | 2(`improve-dev-scripts` + `numbering-rule`) | 5(再加 `bootstrap-dqn-baseline` + `establish-project-lifecycle-spec` + `gemma-reward-generator`) |
| **可視化** | 無 | pygame `play.py` 可載入 model 看 agent 飛 |
| **安全** | `.env` 未在 `.gitignore`(漏洞) | `.env` + `.env.local` 都在 ignore,API key 保護到位 |

---

## 三大成果一句話描述

1. **DQN Baseline 站穩**:vanilla DQN(MLP 64-64, lr 5e-4)在 LunarLander-v3 收斂於 ep 399,比 IJRPR 2025 文獻的 ~1200 ep 快 3 倍 ── 後續所有實驗的底盤
2. **治理 spec 落地**:用「3 個 subagent 平行討論 + 1 個 reviewer 整合」的 multi-agent 模式產出 5 個 capability spec(文件 / 環境 / 實驗 / 評估 / 交付),後續每個 change 都要引用
3. **Gemma reward generator 通線**:Google Gemma 4 31B → 7-arg Python reward → AST 沙箱 + 3-retry → DQN 訓練,end-to-end smoke 全綠

---

## 詳細紀錄索引

| 檔案 | 主題 |
|---|---|
| [02-baseline實作.md](02-baseline實作.md) | DQN baseline change 的完整過程(skeleton → 收斂結果) |
| [03-白話架構文件.md](03-白話架構文件.md) | `白話架構介紹.md` 的設計理念(籃球教練比喻) |
| [04-治理spec多agent協作.md](04-治理spec多agent協作.md) | 3+1 agent 模式建立 `establish-project-lifecycle-spec` |
| [05-Gemma實作.md](05-Gemma實作.md) | `gemma-reward-generator` change + Gemma 第一次寫的 reward |

---

## 重要決策摘要(寫進治理 spec 的)

| 決策面 | 結論 | 理由 |
|---|---|---|
| 環境管理 | **uv 為主、pip+venv 備援** | 安裝快 10-100×,跨平台一致 |
| Python 版本 | `>=3.11,<3.12` | 3.12 改了 ast 模組,會影響 ast-buffer-manager |
| CUDA wheel | `torch==2.5.1+cu121` 鎖定 | 4090 在 cu121 已驗證 |
| 註解語言 | **英文 only, WHY-only** | 程式碼 identifier 是英文,混排不一致 |
| 中文用途 | README、proposal、handover、白話介紹 | 受眾是台灣團隊 + 教師 |
| Seed 數 | 每 condition **5 seeds [42-46]** | n=5 + Mann-Whitney 平衡 power 與 4090 throughput |
| 訓練預算 | **固定 1500 ep,禁用 early-stop** | 公平的 converge_episode 比較 |
| Hermes 外層迴圈 | **5 次 LLM 重寫** | EUREKA 觀察:3 次太少,10 次爆 compute |
| 統計檢定 | **Mann-Whitney U + 5000 bootstrap CI** | n=5 不能假設常態 |
| Win 條件 | **p<0.05 AND ≥10% 差 AND CI 不重疊** | 三條件齊全才算贏,防 cherry-picking |
| Primary metric | **converge_episode**(sample efficiency) | 本專案賣點是「LLM reward 讓 DQN 學更快」 |
| Outlier 處理 | **全部 report,不剔除** | 學術誠實 + DRL 本來就 noisy |
| 4090 共享 | **GitHub Issues 預約 + nvidia-smi 守門** | 雙層協議 |
| Reward 沙箱 | **AST 黑名單 + builtins 白名單**,不防對抗 | LLM 是合作者非攻擊者 |

---

## 主要技術 caveat / 待解問題

1. **LLM reward 的 episode return 不能直接跟 baseline 比** ── Gemma 加了 shaping(中心 / 直立 / 終局放大),`episodes.jsonl["return"]` 是 shaped 值,baseline 是 env-native 值。`closed-loop-fitness` change 必須加「同時記錄 env_reward 與 shaped reward」才能 apples-to-apples
2. **單 seed 的 baseline 結果可能是好運** ── seed=42 收斂 399 ep,但文獻典型 ~1200 ep。正式比較要跑滿 5 seeds(spec 已規定)
3. **Gemma temperature 未設定** ── 預設 ~0.7,結果非完全可重現。`closed-loop-fitness` 階段需決定固定 vs 多樣性
4. **CI 尚未建立** ── 目前靠 pre-push hook 與 reviewer 人工
5. **uv 遷移未做** ── 治理 spec 規定用 uv,但目前還是 pip + requirements.txt;migration 排在後續 change

---

## 後續路線圖(治理 spec 規定的順序)

1. ✅ `bootstrap-dqn-baseline` (本 session 完成)
2. ✅ `establish-project-lifecycle-spec` (本 session 完成)
3. 🔄 `gemma-reward-generator` (本 session 7/9 task group 完成,1500-ep run 正在跑)
4. ⏳ `hermes-memory-layer` — 4 層記憶(SQLite FTS5 為主)
5. ⏳ `ast-buffer-manager` — AST diff + replay buffer 處理
6. ⏳ `closed-loop-fitness` — 7 步閉環 + 多輪迭代 + 跑全部 6 baseline × 5 seeds

每一份 change 結束時都是**獨立可發表的命題**:
- ② Gemma 證明「LLM 寫的 reward 能讓 DQN 收斂」── EUREKA 開源重現
- ③ 記憶證明「有記憶的 LLM 比沒記憶更快收斂」── 新貢獻
- ④ AST/Buffer 證明「換 reward 時保留有用經驗比清空好」── 對應災難性遺忘
- ⑤ 整合 + 統計 = 論文 Section 4

---

## Memory(我下次會自動記得的)

存在 `~/.claude/projects/.../memory/`:

- `project_team_context.md` ── 這是 4 人團隊作業,scoping 要考慮 ownership 與口試答辯責任
- `feedback_doc_style.md` ── 你要白話文件時的固定樣板:單一主比喻 + 術語翻譯表 + 口試版收尾 + 繁體中文
- `project_lifecycle_spec.md` ── `establish-project-lifecycle-spec` 已建立,46 個 Requirement,未來任何 change 都要引用

下次 session 開新 change 時,我會自動帶這些 context。
