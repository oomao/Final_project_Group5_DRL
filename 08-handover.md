# Handover Document (08) - 2026-06-01

## Summary of Changes

從 07（專案已交付）延伸出一份**研討會口頭報告**用的新簡報（有別於先前的影片版）。
聽眾為專精 DRL 的指導老師，敘事策略：**結論先行 + 數據直覺 + 主動揭露限制**。本 session 全程詳見
`aichat_record/方向確立/Hermes-DQN_研討會簡報製作對話紀錄.md`。

- **15 分鐘逐頁口白**：`paper/conference_talk_15min_zh.md`（12 頁、倒敘、含時間表與 Q&A 防禦十題）。
- **三個 HTML 簡報 SKILL 並排比較**（使用者提供）：frontend-slides（zarazhangrui）/ guizang（歸藏，瑞士風）/ huashu（花叔，Pentagram）。各派子代理做一版 → Playwright 截圖對照 → **選定 guizang 瑞士風**（巨數字最直覺）。
- **精修成最終版**：① 修 P6 反轉（四環境全顯示的 ledger-bar）、P9 Part 2（標題不再壓圖）、P3 單位破損；② 依回饋**淺色化**全 deck、**放大字級**（內文 ≥18px）、**白話化**用字、嵌入 demo GIF。
- **重生 P7 關鍵 GIF**：`tools/gen_p7_gif.py` 跑 seed_43 模型，做出「**無記憶落地（+309）vs 有記憶懸停（+15）**」對比，演示 potential-based shaping drift。
- **歸位 + push main**（commit `f82e18c`）：整包簡報放入 `PPT/研討會簡報/`。

## Current Status

- 研討會簡報**完成並在 `main`**。現場用 `PPT/研討會簡報/index.html`（瀏覽器開、GIF 會動），`...瑞士風.pdf` 為靜態備案，`口白_15分鐘.md` + `README.md` 同層。
- `_slide_skills/`（clone 的三個 skill repo + frontend/huashu 兩版 deck + 預覽）**留本機、已加入 `.gitignore`**，未進 repo（避免巢狀 git 與大檔）。
- 專案其餘狀態同 07：論文（EN/ZH）、五段式報告、影片、影片版簡報皆已交付且對齊「獎勵密度假說」。
- OpenSpec：`reward-sandbox-isolation` 仍為**刻意保留的 proposal-only**；其餘 7 個 change 已 archive。`tasks.md` 未變動。

## Next Actions / Open Items

- 沿用 07 未結項目：
  - [ ] **確認「Gemma 4 31B」型號／參數量** — 研討會口白與簡報**也沿用此名稱**，若有誤需連同論文/README/report/兩版 deck 一併更正。
  - [ ] B1 placeholder（非作者重寫人類基準，選配）。
- 本次新增（皆選配）：
  - [ ] 若要 **100% 離線視覺一致**：把 Google Fonts 內嵌進 `index.html`（目前斷網會 fallback 系統字，版面不變、字體略異）。
  - [ ] frontend Signal / huashu Pentagram 兩版若要改用，可再精修（檔在本機 `_slide_skills/_output/`，未進 repo）。
  - [ ] MP4 備案 / 包成 .pptx（本次使用者選擇不做）。
  - [ ] 移除影片版 `PPT_最終版` 的 NotebookLM 浮水印（07 既有項）。

## Notes

- **現場呈現**：整包 `PPT/研討會簡報/` 帶著 → Chrome/Edge 開 `index.html` → F11 全螢幕 → ←→ 翻頁；`B`=靜態、`ESC`=索引。GIF 自動循環，不用點。
- **GIF 管線**：`tools/generate_presentation_gifs.py`（B0 vs Hermes、2×2 grid、3 變體）+ `tools/gen_p7_gif.py`（P7 記憶有害對比）。需 `runs/final/` 模型 + torch/gymnasium（皆在本機可跑）。
- 同 07：`02-ending.sh` 的 handover 為寫死樣板，本 08 亦為**手動撰寫**以反映實際工作。
- **commit 署名**：本 session 的 commit 含 `Co-Authored-By: Claude` trailer；07 記載先前 commit「皆無 Claude 署名」。團隊若要統一（去除署名），日後 commit 可省略此 trailer——已推上 `main` 的不另行改寫（避免 group repo force-push）。
