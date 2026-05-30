# skill 審稿與排版修正

> 本段為本 session 前半的對話紀錄

## 用 skill 審稿

**使用者**：可以試著用這個 skill 修正我們的論文嗎 我覺得還是有點問題（附 GitHub：Master-cai/Research-Paper-Writing-Skills）

**處理**：
- clone 該 skill（彭思達論文寫作筆記改編）,讀 `SKILL.md` + 各 section guide + `paper-review.md`（對抗式審稿清單）
- 對中英兩版做「claim-evidence 對齊」審稿
- **重要發現**：`paper/latex_build/*.tex` 是 `build.py` 用 pandoc 從 `paper/hermes_dqn_paper_{en,zh}.md` **重新生成**的,直接改 `.tex` 會被覆蓋 → 真正的源頭是 `.md`

**抓到並修掉的 §6.1 內部矛盾**（都用論文自己的數據對齊）：
1. 「DQN never reliably solves any of these」但括號寫 Acrobot 成功率 62% → 矛盾
2. 「even Gemma's first attempt recovers most of the gap」對 CartPole 不成立（B2=187.64,僅回收約 18%）
3. 「the gap is modest (CartPole 187.64 → 334.44)」→ 不可能叫 modest,且下一句說「記憶在 CartPole 貢獻最大」自相矛盾

**其他修正**：
- 「vanilla」一詞在摘要/Part 1 指 B0 原生獎勵基準,在 Part 2 指 vanilla DQN 架構 → 摘要/結論統一改「環境原生基準 (B0)」
- 摘要變異性指紋：原文像「稀疏→穩定、密集→高變異」,但 CartPole 稀疏卻 std=113 → 改「分界是塑形空間豐富度而非密度」,與 §6.3 一致

修完後 rebuild 確認 `*_main.tex` 內舊矛盾字串 0 筆、新字串各 3 筆。

## 圖片編號錯亂

PDF 每個圖標題都雙重編號且不一致（如「Figure 2: Figure 6. CartPole…」）—— caption 在 markdown alt-text 內硬寫「Figure N.」,LaTeX 又自動編號,內文「Figure 6」其實指向 PDF 標的「Figure 2」。

**處理**（使用者選「自動編號（建議）」）：移除硬編號、讓 LaTeX 依出現順序自動編,內文所有「Figure N」參照重新對應（架構=1、箱型圖=2、變異=7、逐迭代=8…）。

## 去粗體

**使用者**：改中英文 舉個例子 正常論文不會有這種深色字吧（附第 2 頁截圖,整段粗體小標 + 整句粗體的核心問題）

**處理**：寫腳本掃全文 →
- 跑版小標（Roadmap. / Primary metric. / 各 limitation 條…）→ **斜體**
- 整句粗體（核心研究問題）、行內強調詞（Hermes-DQN、variance signature、B0/B1…）、16 條參考文獻標題 → **一般文字**
- 表格內 highlight（最佳列、WIN、顯著值）→ 保留

EN 斜體化 28、去粗 39;ZH 斜體化 29、去粗 34;驗證殘留粗體只剩表格/標題/作者列。

## 收緊空白

**使用者**：一起收緊

**處理**：`build.py` 兩版 preamble 加 `\usepackage{enumitem}` + `\setlist{topsep=2pt, partopsep=0pt, …}` 收清單上下間距、`\LTpre/\LTpost` 從預設 ~12pt 降到 3/6pt 收長表格間距。英文版頁數 14 → 13。
