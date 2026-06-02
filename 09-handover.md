# Handover Document (09) - 2026-06-02

## Summary of Changes

從 08（研討會 guizang 瑞士風淺色版）延伸，本 session 把這份網頁簡報**改框架為「深度強化學習 Final Project 期末專題報告」**，並比照老師最滿意的參考簡報（碩士口試式）重構內容呈現方式，定版為 **18 頁**。全程詳見
`aichat_record/方向確立/Hermes-DQN_研討會簡報製作對話紀錄.md`（已補階段七）。

- **參考老師範例 `PPT參考.pptx`**（碩論口試，38 頁）的「內容呈現方式」：五章（Motivation / Related Works / Proposed Method / Experimental Results / Conclusion）＋ Outline ＋ 每頁章節標頭「Ch.X 章名 (n/N)」＋ 大頁碼 ＋ 就地引用 [N] ＋ 一頁一重點 ＋ 結尾 Contributions / Limitations / Future works。
- **五章重構**：原「結論先講」12 頁倒敘 → 傳統依序講；精簡成 16 頁，後擴為 18。
- **淺色化收尾**：研究缺口黑卡、記憶 Working 層等殘留深色塊改淺灰；引用行 `max-width:73vw` 避免撞翻頁提示。
- **Reframe 研討會 → Final Project**：封面 chrome、README、口白標題改「深度強化學習 期末專題」。
- **封面英文化**：大字 Hermes-DQN ＋ 英文主標「Memory-Augmented LLM Framework for Automated RL Reward Design」＋ 中文小副標。
- **系統方法放論文架構圖** `fig1_architecture.png`。
- **Ch.4 每個結果都配 demo GIF（GIF 左＋數據右一致版面）**：結果一 `grid_2x2`、結果二 `p7_memory_harm`、結果三 **新重生** `variance_fingerprint`（`tools/gen_variance_gif.py`：MC 三 seed 一致 vs LL 三 seed 分歧）、結果四 `dqn_compare_cp`；並新增 **結果總表**（六條件 × 四環境 env-native mean）。
- **結尾加 Thank You / Q&A 頁**；定版 18 頁，頁碼全重編、Ch.4 變 (n/6)。
- **理性用字**：移除 ★／全場重心／解鎖／「只在…才」等浮誇字眼（同步改口白）。
- **口白同步 18 頁版**（時間表約 16 分、含章節轉場與 Q&A 防禦）。
- **Push（commit 1505d3a）**：依使用者要求，**範例 `PPT參考.pptx` 與口白稿不上傳**（gitignore + 從 repo 移除、本機保留）。

## Current Status

- Final Project 簡報定版 **18 頁**、在 `main`。現場用 `PPT/研討會簡報/index.html`（瀏覽器、GIF 會動），`...瑞士風.pdf` 為備案。
- repo 內 `PPT/研討會簡報/` 僅含 `index.html` + `images/` + `assets/` + PDF + `README.md`；**口白（`口白_15分鐘.md`、`paper/conference_talk_15min_zh.md`）與範例 `PPT參考.pptx` 為本機保留、不在 repo**（已 gitignore）。
- 結構：五章 + Outline + 章節標頭 + 大頁碼 + 就地引用 + 四結果各配 GIF + 結果總表 + Thank You；全淺色、字級放大、理性用字。
- `_slide_skills/`（clone 的 skill repo + frontend/huashu 兩版 deck）仍本機、gitignore。
- 其餘同 07/08：論文、報告、影片、影片版簡報皆已交付且對齊「獎勵密度假說」。

## Next Actions / Open Items

- 沿用未結：
  - [ ] 確認「Gemma 4 31B」型號（簡報／口白沿用此名，若有誤需一併更正）。
  - [ ] B1 placeholder（非作者重寫人類基準，選配）。
- 本次新增（皆選配）：
  - [ ] 資料夾 `研討會簡報` 與 PDF 檔名 `Hermes-DQN_研討會_…` 仍含「研討會」字樣（內容已 reframe 為期末專題）；要的話可改名「期末專題」。
  - [ ] 口白只從 `main` 的 HEAD 移除；**舊 commit 歷史仍存在**。要完全抹除需 rewrite + force-push（group repo，須先協調）。
  - [ ] 100% 離線視覺一致可內嵌 Google Fonts。
  - [ ] frontend Signal / huashu Pentagram 兩版（本機 `_slide_skills/_output/`）若要改用可再精修。

## Notes

- **現場呈現**：整包 `PPT/研討會簡報/` 帶著 → Chrome/Edge 開 `index.html` → F11 → ←→；`B`=靜態、`ESC`=索引。GIF 自動循環。
- **GIF 管線**：`tools/generate_presentation_gifs.py` + `gen_p7_gif.py`（記憶有害對比）+ `gen_variance_gif.py`（變異指紋）。需 `runs/` 模型 + torch/gymnasium。
- 同 07/08：handover 為手動撰寫；commit 依團隊慣例未加 Claude 署名。
