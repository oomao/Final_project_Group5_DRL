# PPT 重製（python-pptx）

> 本段為本 session 的直接對話紀錄

**使用者**：你能幫我生一個PPT嗎 類似風格的有辦法做嗎 你去找PPT模板幫我做這樣

**使用者（範圍／風格選擇）**：
- 範圍 = 重製全部 30 頁 **＋ 補上覺得缺漏的頁**
- 風格 = 維持白底、**有筆記感**的極簡風

---

## 技術選擇

- **建構**：python-pptx（本機已安裝；Node / pptxgenjs 亦可但無需）
- **字型**（確認本機皆有）：Georgia（英文襯線標題）、Microsoft JhengHei（中文）、Consolas（程式碼）
- **QA 轉檔**：本機無 LibreOffice / pdftoppm → 改用已安裝的 **PowerPoint COM** 轉 PDF，再用 **PyMuPDF** 轉圖目視檢查
- 從論文撈出**真實數據**（表 1–6、超參數、環境表、統計方法、seed_43 案例）確保圖表正確

---

## 設計系統

- 白底（#FFFFFF）、章節大數字頁、標題不加底線（避免 AI 味）
- 母題：黃色便利貼（旋轉）＋螢光標記＋綠(正向)/紅(警告)標註框
- Georgia + JhengHei 混排（latin / ea 分別設定東亞字型）

---

## 34 頁結構

- 忠實重製原 30 頁內容與順序
- **新增 4 頁**：實驗設定、評估指標＋統計方法、AST 差異如何計算、完整結果數據表（附錄）
- 限制頁從 4 條補成論文版 **6 條**（含「密集環境僅一個／密度-塑形混淆」）

## 修正

- 錯字：種破→效能提升、原最策略→原始最優策略、上一輪輪→上一輪
- 移除 NotebookLM 浮水印
- 緩衝策略統一為 4 種（KEEP / PARTIAL_KEEP / DECAY / CLEAR）
- 「Gemma 4 31B」→ 中性「開源 LLM」（避開無法查證的型號）

---

## 建構過程的 bug

| 問題 | 修正 |
|---|---|
| `_font() got unexpected keyword '_sa'` | `shape_text` 先把段落層級的 `_sa`/`_line` pop 掉，再傳給字型函式 |
| `ChartFormat has no attribute 'fore_color'` | 圖表上色改用 `.format.fill.fore_color.rgb` |

## QA（render全 34 頁目視檢查）

發現並修正 2 處重疊：
- **S12**：右上「Closed-loop…重複 5 次」說明壓到方塊頂 → 併入底部說明
- **S19**：右下「總規模／120 次」便利貼壓到 B3-hermes-full 列 → 壓縮列距、便利貼移到表格下方

文字檢查：`種破` / `原最` / `上一輪輪` / `NotebookLM` / `Gemma 4 31B` 計數皆為 0。
