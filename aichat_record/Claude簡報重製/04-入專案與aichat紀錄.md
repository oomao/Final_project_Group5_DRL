# 入專案與 aichat 紀錄

> 本段為本 session 的直接對話紀錄

## 放進專案

**使用者**：可以把這些檔案放到我們專案裡面嗎 不要再下載

**處理**：
- 專案已有 `PPT/` 資料夾，命名慣例為 `PPT_第N版`（已有第一版、第二版）→ 新檔命名 **第三版**
- 確認 `.gitignore` 未排除 `PPT/`（且已納入 git 追蹤）
- **`build_deck.py` 輸出路徑改為 `PPT\PPT_第三版.pptx`** → 以後重建不再進 Downloads
- 重建到 `PPT/`、用 PowerPoint 產生 `PPT_第三版.pdf`、複製 `build_deck.py` 進 `PPT/`
- 刪除 Downloads 暫存（`Hermes-DQN_v2.*`、`_hermes_slides/`），保留使用者原始 `Hermes-DQN.pdf`

最終 `PPT/`：

```
PPT_第一版.pdf
PPT_第二版.pdf
PPT_第三版.pptx   ← 可編輯（34 頁，132KB）
PPT_第三版.pdf
build_deck.py
yt口白第一版.txt
yt口白第二版
```

附帶好處：第三版為**原生向量簡報**（文字可選取／可改），約 132KB，比前兩版的圖片輸出小約 300 倍。

---

## aichat 紀錄

**使用者**：把我們這邊 討論PPT的部分也加入aichat

→ 建立 `aichat_record/Claude簡報重製/`，含 01–04 四個紀錄檔。

---

## 下一個 Session 目標

- 把投影片的「開源 LLM」改回**正確的模型型號**（若 Gemma 4 31B 屬實則填回；否則同步更正論文與簡報）
- 更新 **YouTube 口白稿**（`PPT/yt口白*`）以對應第三版內容與「記憶密度假說」新發現
- （可選）產出**英文版簡報**
