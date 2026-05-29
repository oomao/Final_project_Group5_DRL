# Handover Document (07) - 2026-05-29

## Summary of Changes

從 06（4 環境實驗進行中）到**專案交付完成**。期間橫跨兩個收尾 session
（`完稿與收尾`、`簡報重製`），中間未寫 handover，本份一併補上。專案目前
**已完稿並全數推上 `main`**：論文（EN/ZH）、五段式報告、簡報（最終版＋可編輯
重製版）、系統介紹影片，且所有文件已對齊最終論文的「獎勵密度假說」。

### Phase A — 完稿與收尾（前一 session，詳見 `aichat_record/Claude完稿與收尾/`）

- 補完 MountainCar + Acrobot → 完整 **4 環境 × 6 條件 × 5 seed = 120 runs**
- **Part 2 DQN 變體泛化**（Double + Dueling，80 runs）→ 模型無關性
- 論文 §4/§5、EN/ZH PDF（NeurIPS 格式）、fig1–8、4 個 demo GIF
- `paper/final_report.md`（教授五段式格式）
- 作者去匿名 → 真實姓名（陳盛茂・林仙安・辛語柔・陳冠宇 / 國立中興大學 資管）
- 中文 PDF 字體：標楷體 + Times New Roman
- README 加入結果表、GIF、論文連結

### Phase B — 簡報重製＋最後確認（本 session，詳見 `aichat_record/Claude簡報重製/`）

- 檢視 NotebookLM 30 頁簡報 → 以 python-pptx 重製 **34 頁 `PPT_第三版`**（修錯字、補缺漏頁、限制頁補成 6 條）
- 團隊自製最終 Canva 版 → 入專案為 **`PPT/PPT_最終版.pdf`**（35 頁）
- **README 最後確認抓到重大問題**：README 與 `final_report.md` 的實驗結果為**過時數據**，且與最終論文**相反**（舊：密集 LunarLander「Hermes 顯著超越」；論文：密集環境記憶**有害** −38.3%、p=0.0317）。已修正三份文件對齊論文：
  - `README.md`（Part 1 總表＋主結果、Part 2、密度假說）
  - `paper/final_report.md`（§3.4 AST、§3.5 網路 64×64、§3.6 條件名、§4.2 超參、§4.4/§4.5 結果、§5 結論、摘要；原本 §4.5 與 §5 還自相矛盾）
  - `hermes_dqn/README.md`（保留誠實的 n=1 pilot，加註指向最終結果）
- `en_body.tex`/`zh_body.tex` 作者去匿名（先前未提交）
- README 影片區新增**最終版影片** `eUtwae1XG4Q`（置於舊版前）
- 全數推上 main（4 個 commit，皆無 Claude 署名）

## Current Status

- **專案實質上已完成並交付**。論文、報告、簡報、影片皆在 `main`。
- 所有文件現在講**同一個一致的故事**：獎勵密度假說
  （稀疏 +31.5%～+116.1%；密集環境記憶 −38.3%、p=0.0317）。
- 交付物位置：`paper/`（PDF + md + figures + gifs）、`PPT/`（第一/二/三版 + 最終版 + build_deck.py）、`README.md`。
- OpenSpec：`reward-sandbox-isolation` 為**刻意保留的 proposal-only**（L3 容器級隔離；
  觸發條件未達，不實作、不歸檔）；其餘 7 個 change 皆已 archive。

## Next Actions / Open Items

- [ ] **確認「Gemma 4 31B」型號／參數量** — 全程未查證，一致出現在論文(.tex/.pdf)、README、final_report、deck。若有誤需全面更正並重編 PDF。
- [ ] **B1 placeholder** — 四環境的 B1 獎勵仍為作者自撰佔位；若需決定性人類基準，依 `evaluation-criteria` spec 應由非作者隊友重寫（論文已將此列為限制，故為選配）。
- [ ] （選配）簡報英文版 / 影片口白稿更新（`PPT/yt口白*`）。
- [ ] （選配）移除 `PPT_最終版` 的 NotebookLM 小浮水印（需從原始檔重新匯出）。

## Notes

- `runs/final_v2/` 仍保留部分 LL 重做資料（論文 §6 以「partial replication consistent with main result」帶過）。
- **`02-ending.sh` 的 handover 是寫死的樣板**（會產生無意義的「startup/ending scripts」內容）；本 07 為手動撰寫以反映實際工作。日後可考慮修補腳本。
