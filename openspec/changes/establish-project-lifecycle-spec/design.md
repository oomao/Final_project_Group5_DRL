## Context

Hermes-DQN 是 DRL 課程的 4 人期末專案,目標是在 NVIDIA RTX 4090 / Windows 11 / Python 3.11 環境下,於 LunarLander-v3 上達成 `mean_reward_last100 ≥ 200 AND success_rate ≥ 0.90`,並透過閉迴路(Gemma reward → Hermes 記憶 → AST/Buffer 管理 → DQN 訓練 → fitness 回饋)展示 LLM 輔助 RL 的提案。

目前狀態:
- `bootstrap-dqn-baseline` 已建立 vanilla DQN 訓練管線,跑得起來、能寫 `runs/<ts>/`,程式骨架到位
- 還有 4 個 feature change 排隊:`gemma-reward-generator`、`hermes-memory-layer`、`ast-buffer-manager`、`closed-loop-fitness`
- 4 人分工尚未在 spec 內白紙黑字,文件/環境/實驗/評估各自的細節長在不同人腦中
- 期末需要交:論文、簡報、demo 影片、口試,且共用「第 N 版」版本號

本 change 解決的是「治理面」的問題:把後續 4 個 feature change 共用的規範一次性 spec 化,讓每個 change 在 proposal 階段就能引用具體 scenario,避免到 PR 階段才在留言區辯論。

## Goals / Non-Goals

**Goals:**
- 為 5 個治理面向(文件、環境、實驗、評估、交付)各自產出 1 份 capability spec
- 每個 spec 內的 Requirement 都附 ≥ 1 個可被 reviewer 機械化檢查的 Scenario
- 統一 3 個討論代理人之間的衝突答案,理由全部記錄在「Decisions」段
- 為 4 個未來 change 提供可被 `openspec validate --strict` 通過的引用基準

**Non-Goals:**
- 不寫 `gemma-reward-generator`、`hermes-memory-layer`、`ast-buffer-manager`、`closed-loop-fitness` 的任何實作
- 不遷移 `requirements.txt` → `pyproject.toml`(這由 env-setup 引用 change 完成)
- 不調整 `bootstrap-dqn-baseline` 既有 spec
- 不訂課程交件日期(由教師公告為準,不寫死)
- 不挑論文發表的會議/期刊(僅規定 IEEE 雙欄格式)

## Decisions

### A. 文件規範 (`doc-standards`)
**頭條決策:** OpenSpec 四件套硬性化、handover 每次 ending 必寫、註解 WHY-only 且英文、白話介紹隨架構同步、PR 必須附 `openspec validate --strict` 輸出。

**收斂的衝突:**
- 註解語言 → **英文 only**。原因:程式碼識別碼/檔名本來就英文,註解中英混排會讓 ruff/IDE 抓不到拼字,且論文/簡報受眾偏國際。中文敘事留給 README、proposal、handover。
- design.md 是否每個 change 都要 → 採「條件性必要」:引入新外部相依、或新跨子系統介面、或 capability ≥ 2 時必要,其餘可省。原因:純文件 change 寫 design.md 是儀式性負擔。

**Alternatives considered:**
- 全面強制 design.md:被否決,會讓 doc-only change 浪費時間。
- 中英混合註解:被否決,跨工具一致性差。

### B. 環境設置 (`env-setup`)
**頭條決策:** Python 3.11.x + cu121 + torch 2.5.1+cu121 鎖定;**主要工具採用 uv**,pyproject.toml + uv.lock 為單一真實來源;pip + venv + requirements.txt 留作 documented fallback(uv 不可用時)。

**收斂的衝突:**
- 環境管理工具 → **採用 uv**。雖然 `bootstrap-dqn-baseline` 目前是 pip + venv,但 uv 同時相容 pyproject.toml 標準,且 `uv export -o requirements.txt` 可自動產生 pip 用的相容檔案。原因:uv 安裝速度比 pip 快 10-100×、跨平台一致、單一 binary 無虛擬環境啟動腳本差異,對 4 人團隊收益最大。Fallback 路徑寫在 spec 中,避免 uv 學習曲線阻斷任何一位隊員。
- 4090 共享協議 → **GitHub Issues 預約 + `nvidia-smi` 啟動時守門,兩層並行**。Issues 處理人類層級的時段協調,`nvidia-smi` 處理「忘了開 Issue 就硬跑」的技術防線。

**Alternatives considered:**
- 維持 pip + venv:被否決,新隊員 onboarding > 30 分鐘,uv 可降至 < 60 秒。
- 改用 conda / poetry:被否決,4090 cu121 wheel 已驗證走 PyPI index,conda 多一層 channel 風險。
- 只用 Issues 預約(不要 nvidia-smi):被否決,人類會忘事。

### C. 實際實驗 (`experiments-protocol`)
**頭條決策:** 每 condition 跑 5 seed (42–46)、1500 ep 固定預算、禁用 early-stop、Hermes 外層 5 次 LLM 重寫、reward_fn.py + SHA-256 強制存檔、三層 run 目錄。

**收斂的衝突:**
- run 目錄階層 → **混合制**。ad-hoc/dev 訓練(< 300 ep)沿用 `runs/<timestamp>/`,正式評估級 run(1500 ep 入統計)必須是 `runs/<experiment_name>/<condition_id>/seed_<NN>/`。原因:既有 `bootstrap-dqn-baseline` 的 flat timestamp 不破壞,新格式由 `experiments-protocol` 引用 change 在進入評估前統一遷移。
- 5 seed vs 8 seed → **先 5 seed**。原因:n=5 + Mann-Whitney U 在 medium effect 下 power 約 60%,搭配 pilot 先估 effect size;若 pilot 顯示 effect 太小,升級到 n=8 再說(這條 escalation 路徑寫進 risks)。

**Alternatives considered:**
- Adaptive episode budget(收斂後停):被否決,會讓 `converge_episode` 之外的 secondary metric 無法在同一截面比較。
- Flat timestamp 一統:被否決,評估腳本要 glob `runs/<exp>/<condition>/seed_*/` 才好聚合。

### D. 評估結果 (`evaluation-criteria`)
**頭條決策:** 6 condition baseline 集 (B0/B1/B2/B3/B3-no-memory/B3-no-AST)、Mann-Whitney U + 5000 bootstrap CI、Win 三條件 (p<0.05 AND mean diff ≥ 10% AND CI 不重疊)、primary metric = `converge_episode`、outlier 全部呈報不剔除。

**收斂的衝突:**
- Primary metric 是 sample efficiency 還是 final reward → **converge_episode 為主**。原因:本專案的論點是「LLM-shaped reward 加速學習」,sample efficiency 才是核心賣點;final reward 留做 secondary。
- Outlier 處理 → **全部呈報、不剔除**,divergent run 在 footnote 揭露 (`n=5 (1 divergent)`)。原因:DRL 結果本就 noisy,剔除 outlier 會被審閱者質疑 cherry-picking。

**Alternatives considered:**
- t-test:被否決,n=5 不能假設常態,Mann-Whitney U 更穩健。
- 純用 final reward:被否決,本專案賣點是效率。

### E. 文件完成 (`final-deliverables`)
**頭條決策:** IEEE 雙欄 8 章 8–12 頁論文、18–22 張 5 段簡報、4–6 分鐘 demo 影片(YouTube unlisted)、四項交付物共用第 N 版號、每位成員至少 1 份可獨立交付的 artifact。

**收斂的衝突:**
- 論文格式 IEEE vs 學校自訂模板 → **先 IEEE 雙欄**。原因:教師多接受 IEEE,且雙欄頁面壓力會迫使我們把內容收斂、避免冗長。若教師明確指定其他模板,屆時再做格式 conversion。
- 第 N 版升版規則 → **觸發條件 = change archive、教師審閱、重大實驗結果**;單純錯字校正不升版。

**Alternatives considered:**
- 單欄 ACL 風格:被否決,雙欄更符合課程慣例。
- 每位作者自做投影片:被否決,風格不一致會在口試扣分。

## Risks / Trade-offs

- **uv 學習曲線(~30 min)** → mitigation: env-setup spec 內寫 uv quickstart + pip + venv fallback。
- **CI 尚未建立** → mitigation: pre-push hook 過渡;期末前不擋 main 但會 warn;CI 建立排程進入 `improve-dev-scripts` 之後的 change。
- **n=5 power 偏低** → mitigation: pilot 階段先估 effect size,若 < medium 則升級 n=8。
- **Gemma API RPM 配額** → mitigation: 4 人共用 1 把 key,實驗時段交錯(4090 booking 表附時段)。
- **Gemma prompt/response log 可能含 PII** → mitigation: 提交前以 `tools/scrub_logs.py`(未來 change)去除個人標識,但本 change 先把這條 risk 記下。
- **4090 共享** → mitigation: GitHub Issues 預約 + nvidia-smi 守門雙層。
- **教師可能指定不同論文格式** → mitigation: 開學週確認後若需切換,IEEE 雙欄→單欄是純 LaTeX/Word 樣板替換,內容不受影響。
- **指導老師索引論文 reference 來源** → 不在本 change 範圍,由 final-deliverables 引用 change 處理。

## Migration Plan

本 change 是**純治理 change**,沒有資料 / 程式碼遷移。其引用順序為:

1. **(本 change merge 後立即生效)** env-setup → 後續任何 change 在 setup 段都引用其 scenario,例如「依 env-setup R1: 鎖定 Python 3.11」。
2. **(env-setup 之後)** doc-standards → 規範後續所有 change 的 PR 流程、handover 寫法。在 02-ending.sh 加上 `openspec validate --strict` 警告。
3. **(實作 change 開始時)** experiments-protocol → `gemma-reward-generator`、`hermes-memory-layer`、`ast-buffer-manager` 三個 change 在 design.md 引用 condition 三元組與 run 目錄階層。
4. **(實作 change 完成時)** evaluation-criteria → `closed-loop-fitness` 必須引用 6 baseline 集 + Win 三條件 + 七件套 reproducibility。
5. **(期末前 2 週)** final-deliverables → demo-day 倒數時逐項勾選 README 最低狀態、論文章節分工、簡報段落分工。

Rollback:本 change 為純 spec,移除 `openspec/changes/establish-project-lifecycle-spec/` 即可回到當前狀態,無資料損失。

## Open Questions

- **指導老師是否接受 IEEE 雙欄?** → 開學週第一次 office hour 確認;若否,改為單欄等校方模板,內容章節結構不受影響。
- **CI(GitHub Actions)何時建?** → 期末前若有時間,加 `lint + openspec validate` 工作流;沒時間則維持 pre-push hook + PR reviewer 人工檢查。
- **Gemma temperature 設 0 vs > 0?** → 由 `gemma-reward-generator` change 決定;本 change 不限制,但若選 > 0,evaluation-criteria 的可重現七件套必須記錄 sampling seed。
- **是否要在 `aichat_record/` 留 LLM 互動 log 作為 supplementary?** → 不收進論文 supplementary;但專案內部留存做後驗。
- **MIT vs Apache vs 課程內部用授權?** → 預設 MIT(與既有 `pyproject.toml` 預期相容);若課程要求繳交 closed source,改為「Course internal use only」,在 README 最後一節宣告。
