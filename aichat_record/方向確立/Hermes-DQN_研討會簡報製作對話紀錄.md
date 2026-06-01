# Hermes-DQN 研討會簡報製作對話紀錄

> **專案**：Hermes-DQN 研討會口頭報告簡報（有別於先前的影片簡報）
> **日期**：2026-06-01
> **參與者**：Mao × Claude
> **產出**：15 分鐘逐頁口白、三個 HTML 簡報 SKILL 並排比較、選定 guizang 瑞士風並精修成最終版、重生一支 P7 關鍵 demo GIF
> **最終檔案**：`PPT/研討會簡報/`（index.html + images/ + assets/ + PDF + 口白）

---

## 概述

先前已有一份「影片用」的簡報（`PPT/PPT_最終版.pdf`，NotebookLM 手繪／白底便利貼風，由 `PPT/build_deck.py` 產生）。本次目標是另做一份**研討會口頭報告**用的簡報，時長約 15 分鐘，台下是專精 DRL 的指導老師，要求「以最簡單的方式呈現最數據的直覺」。

整個過程分六個階段：① 口白製作 → ② 三個 SKILL 比較 → ③ 選定 guizang 並初步修正 → ④ 淺色化／放大字級／白話化／嵌 GIF → ⑤ 重生 seed_43 懸停 GIF → ⑥ 現場呈現方式與檔案歸位。

---

## 階段一 · 15 分鐘逐頁口白

先讀通論文（`paper/hermes_dqn_paper_en.md` / `_zh.md`）與舊影片簡報脈絡，確立研討會版與影片版的差異：

- **影片版**：偏「我們在做什麼、為什麼做」——三大痛點 → 架構 → 資料流 → 消融設計，鋪陳動機與方法。
- **研討會版**：聽眾懂 DRL，不需解釋基礎；要**結果先行**，把最反直覺的負向結果當開場鉤子。

產出 `paper/conference_talk_15min_zh.md`：12 頁、倒敘結構、每頁標「數據直覺」一句話、附時間分配表與 **Q&A 防禦準備**（n=5、短 budget、B1 placeholder、density vs richness 混淆等十題預備答案）。

核心敘事：**記憶不是萬靈丹，效益取決於 reward density**。稀疏獎勵環境記憶幫忙（+31%~+116%）；密集獎勵環境記憶**統計顯著有害**（−38.3%，p=0.0317）——這條研究線第一個顯著負向結果。

---

## 階段二 · 三個 HTML 簡報 SKILL 並排比較

使用者提供三個 GitHub 簡報 SKILL，要求各做一版比較效果：

| SKILL | 來源 | 定位 | 產出 |
|---|---|---|---|
| **frontend-slides** | zarazhangrui | 零依賴單檔 HTML、1920×1080 固定舞台、34 套 bold 模板 | Signal 模板（海軍藍＋金，期刊式內斂） |
| **guizang-ppt-skill** | 歸藏 op7418 | 單檔橫向翻頁、WebGL 背景、Motion 動效；A 電子雜誌／**B 瑞士國際主義** | 瑞士風（IKB 藍、巨數字、資料驅動） |
| **huashu-design** | 花叔 alchaincyf | HTML 高保真／可帶旁白動畫、設計顧問模式 | Pentagram 資訊設計（黑白報紙＋紅色語意） |

作法：clone 三個 repo → 各自讀通 SKILL.md 與模板 → 寫一份共用內容大綱 `_slide_skills/DECK_BRIEF.md` → **並行派三個子代理**各做一個 12 頁 HTML 版 → 用 Python Playwright（system Chrome）統一截圖 → 做並排對照圖。

三版都把關鍵頁（P2 鉤子、P7 記憶有害 248→153）做出來。使用者偏好「最簡單呈現數據直覺」，最終**選定 guizang 瑞士風**（一頁一個巨大數字最直覺）。

> 三個 clone 的 repo 與另兩版 deck 留在本機 `_slide_skills/`（已加入 `.gitignore`，不進 repo）。

---

## 階段三 · 選定 guizang 並初步修正

逐頁檢視後修掉三個問題：

| 頁 | 問題 | 修正 |
|---|---|---|
| P6 反轉 | MountainCar／LunarLander 沒顯示數值、`bar-grow` 動畫讓長條沒展開、下半空白 | 改成可靠的 ledger-bar（仿 P8）：四環境全顯示，藍色 WIN ×3＋灰色「反轉」×1，長條按 \|Δ\| 比例 |
| P9 Part 2 | 黑色標題框絕對定位壓在 fig7 上、遮住第一張子圖 | 標題移到圖上方獨立帶狀，fig7 完整白底呈現，下方三 KPI |
| P3 背景 | 末格「研究缺口 0」單位文字破損 | 改「0／從未經多環境＋統計檢定驗證」 |

並把翻頁提示由簡體（翻页/静态）改繁體（翻頁/靜態/索引）。Swiss 驗證器 `validate-swiss-deck.mjs` 通過 12 頁。

---

## 階段四 · 淺色化／放大字級／白話化／嵌入 GIF

使用者回饋：**更喜歡淺色系、字體要大（台下看得清）、用字更白話、放 demo GIF**。派子代理一次到位、自我渲染驗證：

1. **淺色系**：封面（藍底→白底墨字＋藍「獎勵設計」）、P2 鉤子左半（黑→白）、P4／P7（黑→白）、P11 底部黑條、P12 結尾（藍→白）。全部 inline 白字／rgba 白翻成 ink/secondary/helper，IKB 藍只當重點色。無任何深色背景頁。
2. **放大字級**：內文 ≥18px、圖註 ≥16px、標籤 ≥14px；淺灰小字改深色提高對比；p 值、「數據直覺」、卡片說明全部放大。
3. **白話化**：例如 P4 標題「記憶與 buffer 都圍著公平評估這個錨點轉」→「三個模組，共用同一把公平的尺」；P5「六個條件，鎖定記憶那一組乾淨對照」→「六個條件，只比『有沒有記憶』」。數字、p 值、技術名詞全保留。
4. **GIF**：封面放四環境 `grid_2x2.gif`；P7 放重生的 `p7_memory_harm.gif`（見階段五）。P6 因四條長條已填滿，刻意不塞。

---

## 階段五 · 重生 seed_43 懸停 GIF（P7 主角）

確認重生環境齊全（torch 2.5.1 + gymnasium 1.2.3，`runs/final/` 各條件各 seed 的 `model_final.pt` 都在）。原本 `ll.gif` 用的是接近平均的 seed，看不出退化策略。

寫 `tools/gen_p7_gif.py`（沿用 `tools/generate_presentation_gifs.py` 的 rollout 函式）：載入 **B3-no-memory 代表 seed** 與 **B3-hermes-full seed_43 iter_05** 兩個模型，在同一 episode seed 並排 rollout。挑到 seed 7012：**無記憶 +309（穩穩落地）vs seed_43 +15（半空懸停不落地）**。底部標 −38.3%／p=0.0317／seed_43=11.6。

這支 GIF 直接演示論文的機制假說（potential-based shaping drift）：同一套 LLM 獎勵，差別只在記憶——無記憶落地，有記憶卻學會貼地懸停拿小獎勵、不去完成 terminal landing。放在 P7 當全場重心。

---

## 階段六 · 現場呈現方式與檔案歸位

**怎麼讓 GIF 動**：用瀏覽器開 `index.html`（不是 PowerPoint）→ F11 全螢幕 → ←→ 翻頁，GIF 自動循環播放。整包資料夾要一起帶（index.html + images/ + assets/）。PDF 為靜態備案。

**檔案歸位**：把整包簡報複製到 `PPT/研討會簡報/`，與既有影片版 PPT 並存。新增 `tools/gen_p7_gif.py`、`tools/generate_presentation_gifs.py`、`paper/gifs/presentation/*.gif`、`paper/conference_talk_15min_zh.md`，push main。

---

## 附錄 · 最終檔案清單

```
PPT/研討會簡報/
├── index.html                              # 主簡報（現場用，GIF 會動）
├── images/                                 # 8 張論文圖 + cp/grid_2x2/p7_memory_harm GIF
├── assets/                                 # Motion One 動畫
├── Hermes-DQN_研討會_guizang瑞士風.pdf     # 靜態備案
├── 口白_15分鐘.md                          # 逐頁口白 + Q&A 防禦
└── README.md                               # 現場操作說明

paper/conference_talk_15min_zh.md           # 口白正本
paper/gifs/presentation/p7_memory_harm.gif  # 重生的 P7 GIF
tools/gen_p7_gif.py                          # P7 GIF 生成腳本
```

## 附錄 · 關鍵數據

- 反轉：CartPole +116%（p=.032）／MountainCar +31.5%（p=.011）／Acrobot +57.5%（p=.095）／LunarLander −11.4%（p=1.0）
- 記憶有害：LunarLander no-memory 248.77 → full 153.56，−38.3%，p=0.0317；seed_43=11.6（貼地懸停）
- 變異 std：MountainCar 3.08／Acrobot 4.39（穩）｜CartPole 113.18／LunarLander 91.40（亂）
- Part 2：稀疏 9/9 正向、密集三變體皆負、Hermes 跨變體 all p>0.3（與架構正交）
