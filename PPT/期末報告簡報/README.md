# Hermes-DQN 期末專題簡報（網頁版 · 瑞士國際主義風）

15 分鐘口頭報告用。聽眾為專精 DRL 的指導老師，敘事策略：**結論先行 + 數據直覺 + 主動揭露限制**。
由 guizang-ppt-skill（風格 B Swiss International，IKB 藍）生成，已改為**淺色系**、放大字級、白話用字，並嵌入 demo GIF。

## 怎麼報告（讓 GIF 動起來）
1. **整包資料夾**帶著（`index.html` + `images/` + `assets/`），不能只拿 index.html。
2. 用 **Chrome / Edge** 打開 `index.html`（雙擊即可）。
3. 按 **F11** 全螢幕。
4. **← → 或空白鍵**翻頁；GIF 會自動循環播放，不用點。
5. 備用鍵：**B** = 靜態（關動畫，弱投影機更穩）、**ESC** = 縮圖索引跳頁。

> 沒網路也能播（圖／GIF 都在本機）；只有字型走 Google Fonts，斷網會自動換成系統字。
> 最穩做法＝自備筆電接 HDMI。

## 檔案
| 檔案 | 用途 |
|---|---|
| `index.html` | 主簡報（現場用，GIF 會動） |
| `images/` | 8 張論文圖表 + 3 支 demo GIF（**必須**與 index.html 同層） |
| `assets/` | Motion One 動畫（離線備援） |
| `Hermes-DQN_研討會_guizang瑞士風.pdf` | 靜態備案（GIF 顯示第一幀） |
| `口白_15分鐘.md` | 逐頁口白 + Q&A 防禦稿 |

## 18 頁・五章結構（Final Project 報告：依序講，仿老師參考簡報的呈現方式）
1 封面（英文標題）→ 2 Outline → **Ch.1 Motivation**（獎勵設計兩難 / EUREKA＋研究問題）→ **Ch.2 Related Works**（LLM 寫獎勵比較表 / 記憶・buffer・研究缺口）→ **Ch.3 Proposed Method**（論文系統架構 / 四層記憶＋獎勵生成 / AST buffer）→ **Ch.4 Experimental Results**（實驗設計 / **結果總表** / 結果一反轉 / **結果二記憶有害** / 結果三變異 / 結果四模型無關；四個結果各配 demo GIF）→ **Ch.5 Conclusion**（Contributions / Limitations＋Future works）→ **18 Thank You / Q&A**

每頁固定章節標頭「Ch.X 章名 (n/N)」＋大頁碼＋就地引用 [N]；Ch.4 每個結果都有 GIF（GIF 左＋數據右一致版面）；視覺維持 Swiss 淺色。

關鍵發現（Ch.4 結果二）：**記憶在密集獎勵環境「統計顯著有害」（−38.3%，p=0.0317）**——據我們所知這條研究線第一個顯著負向結果。GIF 直接演示：同一套 LLM 獎勵，無記憶穩穩落地、有記憶 seed_43 卻學會貼地懸停。

> `口白_15分鐘.md` 已對齊這份 18 頁五章結構（依序講＋章節轉場＋Q&A 防禦）。
