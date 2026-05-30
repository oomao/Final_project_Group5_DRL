# IEEE 兩欄轉換與收尾

> 本段為本 session 後半的對話紀錄

## 發現格式不符規範

**使用者**：我覺得英文的論文格式還是有點沒符合規範

**處理**：看 PDF（14 頁、單欄）+ 查 `openspec/specs/final-deliverables/spec.md` → 規範要求 **IEEE 兩欄、8–12 兩欄頁**,但 `build.py` 用的是 **NeurIPS 單欄**（`neurips_2024`）。這就是不符規範之處。確認後轉換英文版。

## 英文版 → IEEE 兩欄

- `PREAMBLE_EN` → `\documentclass[conference]{IEEEtran}`、`\IEEEauthorblock*` 作者區、`morefloats`、`\emergencystretch`
- 新增 `ieee_tables()`：pandoc `longtable`（兩欄非法）→ 跨欄 `table*` 真浮動表（有標號的給 `\caption`,小表做無標號跨欄）
- 圖 → 跨欄 `figure*`;閉環虛擬碼太寬 → 包成**無 caption** 的跨欄 `figure*`（無 caption 才不佔圖號,內文 Figure N 才不會位移）
- 附錄長 `\texttt` URL/路徑 → `\url` 才能斷行
- 結果：11 頁、overfull 從 10 降到 1

## 嚴格 IEEE 化（1–3 項）

**使用者**：繼續處理上面 1–3 的嚴格 IEEE 小項

- 內文引用 `(Ma et al., 2024)` → `[1]`…`[16]`（依參考文獻順序）;參考清單 `\labelenumi` 改 `[N]`
- 章節 `Section 2…7` → 羅馬 `II…VII`;子節 `§6.4/§6.2/§3.5` → `VI-D/VI-B/III-E`
- `Figure N` → `Fig. N`;補上圖 3–6 的內文引用
- 驗證：[1]–[16] 全到、author-year 殘留 0、阿拉伯 Section 殘留 0、Figure 殘留 0

## 中文版 → IEEE 兩欄

**使用者**：你覺得中文跟著換會更好嗎 會的話一起換

**判斷**：是,兩份一致更專業、也都符合 spec;但 CJK + IEEEtran 較棘手,且羅馬章節放中文論文怪 → 做語言中性的部分（兩欄、`[N]` 引用、真浮動表）,**章節用阿拉伯**讓「第 4 節」自然。

- `PREAMBLE_ZH` → IEEEtran + xeCJK（xelatex、標楷體 DFKai-SB）
- 阿拉伯章節/子節/表格編號對齊既有「第 N 節 / 6.4 節 / 表 N / 圖 N」;子節要另外覆寫 IEEEtran 的 `\thesubsectiondis`（標題顯示用）才會變阿拉伯
- `cleanup_body` 的 IEEE 轉換改為**中英都套用**（移除 `lang=="en"` gating）;表格 caption regex 同時吃 `Table N:` 與 `表 N:`
- 引用 → `[N]`（中文變體字串,如 `Masadome 與 Harada`、`Ng、Harada 與 Russell`）
- **第一次就編譯成功**,11 頁、CJK 正常
- 順手修英文版瑕疵：表格編號原本羅馬 `TABLE II` 但內文 `Table 2` → `\thetable` 改阿拉伯,現一致

## 收尾

- 兩份都 IEEE 兩欄、各 11 頁、各剩 1 個極小 overfull
- 補上本 aichat 紀錄（`Claude論文修訂與IEEE轉換/`）後,一起 commit + push 到 `oomao/Final_project_Group5_DRL` main
- 註：`paper/gifs/presentation/`、`tools/generate_presentation_gifs.py` 為其他工作的未追蹤檔,未納入本次 commit
