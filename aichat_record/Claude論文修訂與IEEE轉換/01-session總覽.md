# Claude 論文修訂與 IEEE 轉換 — Session 總覽

**時間**：2026-05-30
**對話工具**：Claude Code（FleetView）

## 這個階段做了什麼

本 session 以 GitHub 上的「Research-Paper-Writing-Skills」skill（彭思達論文寫作開源筆記改編）對論文做對抗式審稿與修正,接著把中英兩版從 NeurIPS 單欄全面轉成 **IEEE 兩欄**,並做引用／章節的嚴格 IEEE 化。

### 第一部分：skill 審稿與內容／排版修正（見 02）

- **§6.1 內部矛盾**（claim-evidence 對齊,皆用論文自己的數據對齊）：
  - 「never reliably solves any」與括號 Acrobot 62% 矛盾 → 改「CartPole/MountainCar 解不出、Acrobot 僅不穩定解出」
  - 「B2 回收大部分落差」對 CartPole 不成立（僅約 18%）→ 改為依環境而異
  - 「差距並不大 (CartPole 187.64→334.44)」自相矛盾 → 改「差異甚大」
- **vanilla 用詞衝突**：摘要/結論的「vanilla baseline」與 Part 2「vanilla DQN 架構」混淆 → 統一為「環境原生基準 (B0)」
- **摘要變異性框架**：原本暗示隨密度翻轉,但 CartPole（稀疏）std=113 → 改為「以塑形空間豐富度為分界」
- **圖片編號錯亂**：caption 內硬寫「Figure N.」+ LaTeX 自動編號 → PDF 出現「Figure 2: Figure 6.」雙重且不一致 → 移除硬編號、改自動編號、內文參照全部對齊
- **去除過量粗體**（使用者：「正常論文不會有這種深色字」）→ 跑版小標改斜體、整句/行內強調/參考文獻標題改一般、表格 highlight 保留
- **收緊版面空白**：`enumitem \setlist` 收清單間距、`\LTpre/\LTpost` 收表格間距

### 第二部分：IEEE 兩欄轉換（見 03）

- **英文版**：`build.py` PREAMBLE_EN → IEEEtran 兩欄;新增 `ieee_tables()` 把 pandoc longtable（兩欄不能用）改寫成跨欄 `table*` 真浮動表;圖改跨欄 `figure*`;虛擬碼跨欄;長 URL 改 `\url`
- **嚴格 IEEE 化**：內文引用 → `[N]`、參考清單 → `[N]`、章節 → 羅馬（Section IV / VI-D）、Figure → Fig.、補上圖 3–6 的內文引用、表格編號改阿拉伯對齊內文
- **中文版**：PREAMBLE_ZH → IEEEtran + xeCJK（xelatex、標楷體 DFKai-SB）;阿拉伯章節/子節/表格編號對齊既有中文參照（第 N 節 / N.M 節 / 表 N / 圖 N）;引用 → `[N]`

## 最終狀態

| | 中文版 | 英文版 |
|---|---|---|
| 版式 | IEEE 兩欄 (IEEEtran+xeCJK) | IEEE 兩欄 (IEEEtran) |
| 頁數 | 11 頁 | 11 頁 |
| 引用 | [1]–[16] + 參考 [N] | [1]–[16] + 參考 [N] |
| 章節 | 阿拉伯 (1、6.4) | 羅馬 (I–VII、VI-D) |
| 表格/圖 | 真浮動表 表 1–6 / 跨欄 圖 1–8 | 真浮動表 TABLE 1–6 / 跨欄 Fig. 1–8 |

符合 `openspec/specs/final-deliverables/spec.md` 的「IEEE 兩欄、8–12 兩欄頁」要求。

## 主要異動檔案

| 檔案 | 異動 |
|---|---|
| `paper/hermes_dqn_paper_en.md` | 內容修正 + [N] 引用 + Fig./Section 參照 + 去粗體 |
| `paper/hermes_dqn_paper_zh.md` | 內容修正 + [N] 引用 + 去粗體 |
| `paper/latex_build/build.py` | EN/ZH preamble 改 IEEEtran;新增 `ieee_tables()` 等轉換 |
| `paper/hermes_dqn_paper_en.pdf` | 重編（IEEE 兩欄,11 頁） |
| `paper/hermes_dqn_paper_zh.pdf` | 重編（IEEE 兩欄,11 頁） |
| `paper/latex_build/{en,zh}_{body,main}.tex` | 由 build.py 重新生成 |

## 下一個 Session 目標

- **確認模型型號**：「Google Gemma 4 31B」是否屬實（業界熟知為 Gemma 3 27B）;若筆誤需同步更正論文與簡報
- **「Rainbow 七項組件」**：通常為 6 項擴充,需確認計數
- 參考文獻目前為 author-year 對應出的 `[N]`,未用真正 bibtex/thebibliography（如需可再升級）
- 確認 `docx/` 交付版本是否需與這份 IEEE LaTeX 對齊

---

*詳細對話見各子檔案*
