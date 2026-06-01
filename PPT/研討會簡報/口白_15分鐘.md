# Hermes-DQN 研討會簡報 — 15 分鐘大綱與口白

> **場合**：研討會口頭報告，約 15 分鐘
> **聽眾**：專精 DRL 的指導老師（不需要解釋 DQN / sparse reward / replay buffer 等基礎）
> **設計原則**：結果先行、數據直覺優先、主動面對方法學弱點
> **與「影片簡報」的差別**：影片版鋪陳動機與架構（痛點→方法）；本版把最反直覺的發現當鉤子，方法只當「達成手段」一頁帶過

---

## 設計理念（為什麼這樣排）

1. **倒敘**：第 2 頁就把 punchline（記憶在密集獎勵顯著有害）講完，方法放後面。專家聽眾要的是「你發現了什麼」，不是「你怎麼一步步做」。
2. **每個結果＝一張圖＋一個數字＋一句直覺**。不堆字。
3. **限制自己先講**。n=5、短 budget、B1 placeholder 這些，老師一定會問；先講掉，反而顯得嚴謹、可信。
4. **負向結果是賣點**，不是瑕疵——這是這條研究線第一個統計顯著的「記憶有害」報告。

---

## 時間分配總表（總計 ~900 秒 / 15 分）

| # | 投影片 | 秒數 | 核心 |
|---|---|---|---|
| 1 | 封面 | 20 | 一句話定位 |
| 2 | 結論先講（鉤子） | 70 | 反轉 + 記憶有害 |
| 3 | 背景：EUREKA 與三個假設 | 70 | 為什麼要測 |
| 4 | Hermes-DQN 是什麼 | 90 | 三模組 + 閉環 + 公平評估 |
| 5 | 實驗設計 | 90 | 4 環境×6 條件×n=5，兩組關鍵對照 |
| 6 | 結果一：反轉 | 110 | fig2 headline |
| 7 | 結果二：記憶在密集獎勵有害 | 120 | fig4 + seed_43 |
| 8 | 結果三：變異性指紋 | 110 | fig3 + fig5 |
| 9 | 結果四：模型無關（Part 2） | 70 | fig7 |
| 10 | 限制 | 70 | 自曝弱點 |
| 11 | 結論 + 可證偽預測 | 70 | reward density |
| 12 | 致謝 / Q&A | 10 | — |

---

## 逐頁大綱與口白

### 投影片 1｜封面（20 秒）
**畫面：** 標題「Hermes-DQN：開源 LLM 記憶增強的自動化獎勵設計」＋四環境 demo GIF（`grid_2x2.gif`）當背景＋作者/校名。
**口白：**
> 各位老師好，我們這組做的是 Hermes-DQN。一句話講，就是用開源 LLM 自動幫 DQN 設計獎勵函數；然後我們去測一個大家都假設成立、但其實沒被驗證過的問題——記憶到底有沒有用。

---

### 投影片 2｜結論先講（70 秒）— **鉤子**
**畫面：** 上半 `fig2_headline.png`；下半一行大字「Memory is *not* a universal upgrade.」
**口白：**
> 我直接先講結論，等一下再回頭補方法。
> 這條 LLM 寫獎勵的研究線——從 EUREKA 開始——有一個幾乎沒人質疑的假設：給 LLM 加上跨輪記憶，讓它累積過去成功的獎勵函數，應該會越寫越好。
> 我們把這個假設拿來做 with / without memory 的對照。結果是兩個字：看任務。
> ［指圖］三個稀疏獎勵環境，記憶版的 Hermes 比原生 baseline 高 31% 到 116%。但在唯一一個密集獎勵環境 LunarLander，它不只沒贏——記憶機制本身讓表現顯著掉了 38%，p 等於 0.0317。
> 據我們所知，這是這條研究線上第一個統計顯著的「記憶有害」結果。今天整場，就是要把這個反轉、還有它背後的直覺講清楚。

**數據直覺：** 記憶不是免費升級；它的價值是任務相依的。

---

### 投影片 3｜背景：EUREKA 與三個假設（70 秒）
**畫面：** 左 EUREKA 數字（83% 任務、+52%）；右三條延伸線（開源 / 記憶 / buffer）各一行。
**口白：**
> 快速交代背景，這個聽眾應該都很熟。
> EUREKA 在 2024 證明了 GPT-4 寫的獎勵可以贏過人類專家，83% 的任務、平均加 52%。之後社群往三個方向延伸：一個是把閉源 API 換成開源模型，省成本、可重現；再來是加記憶，讓它跨輪 refine；還有一個是處理 replay buffer——因為獎勵一改，舊的 transition 就失效，會 catastrophic forgetting。
> 這三條線，大家「預設」可以疊在一起、而且只會更好。但這個複合系統到底是不是普遍有效，沒有人用多環境、帶統計檢定的方式測過。我們就是來補這個測試。

---

### 投影片 4｜Hermes-DQN 是什麼（90 秒）
**畫面：** `fig1_architecture.png`，三模組標色（藍=Gemma / 紫=記憶 / 綠=AST buffer）。
**口白：**
> 我們把這三條線組成一個系統，叫 Hermes-DQN，三個模組。
> ［指圖］第一個是 reward author，用 Gemma 4 31B，純開源、本地跑，取代 GPT-4。
> 第二個是四層記憶，借 Nous Research 的 Hermes Agent 分類——Procedural、Semantic、Episodic、Working。每跑完一輪，就把獎勵原始碼加上它的 fitness 存進去；下一輪用 fitness 排序取 top-k，塞回 prompt 當「過去最好的嘗試」。
> 第三個是 AST-aware buffer manager。它用 Python 的 ast 去 diff 新舊獎勵函數，分四類——完全相同、只改函式體、控制流變動、整個重寫——對應四種 buffer 策略：KEEP、PARTIAL_KEEP、DECAY、CLEAR。
> 整個是一個閉環，跑 5 輪。這邊有個對各位很關鍵的點：不管訓練時用哪個獎勵，**評估永遠是用環境「原生」獎勵、在 100 個沒見過的 seed 上跑**。這是唯一公平的跨條件指標。

---

### 投影片 5｜實驗設計（90 秒）
**畫面：** 左 4 環境表（標密集/稀疏、obs/action 維度）；右 6 條件表，框出兩組對照箭頭。
**口白：**
> 實驗設計。
> ［指圖］環境選了四個古典控制，刻意橫跨獎勵密度：LunarLander 是唯一密集的，CartPole、MountainCar、Acrobot 是稀疏的。動作數也涵蓋 2、3、4，避免把密度效應跟動作數搞混。
> 每個環境六個條件。最重要的是兩組對照：**B3-full 對 B0-原生**，回答「這套系統到底有沒有幫助」；**B3-full 對 B3-no-memory**，這兩個唯一差別就是記憶，回答「記憶本身有沒有用」。中間還有 B2 單次 Gemma、B3-no-AST，去拆每個模組的貢獻。
> 每格 n=5 個 seed，全保留、不剔除發散的 run。檢定用 Mann-Whitney U、非參數，配 bootstrap 95% 信賴區間。Part 1 總共 120 次訓練，Part 2 再加 80 次。

---

### 投影片 6｜結果一：反轉（110 秒）
**畫面：** `fig2_headline.png` 放大；四環境 Δ% 與 p 值標在長條上。
**口白：**
> 進到結果。第一個、也是最核心的——反轉。
> ［指圖］這張是 B3-full 對 B0 在四個環境的相對差。三根綠的是稀疏環境，全往上：CartPole 加 116%、p=0.032；MountainCar 加 31.5%、p=0.011；Acrobot 加 57.5%、方向明確但 p=0.095 沒過門檻。一根紅的是密集的 LunarLander，往下，減 11%，而且 p=1.0，完全打平。
> 這邊的數據直覺其實很乾淨：在 CartPole 跟 MountainCar，原生獎勵下 vanilla DQN 的成功率是 **0%**——根本學不起來，LLM 寫的塑形是把一個「學不動」的任務解鎖。但 LunarLander 的原生獎勵本來就密、訊號強，B0 已經有 173 分、接近解題門檻 200，這時候再疊 LLM 塑形，就沒有空間幫了。
> 一句話：**塑形只在原生訊號弱的時候有用。**

---

### 投影片 7｜結果二：記憶在密集獎勵有害（120 秒）— **重頭戲**
**畫面：** 左 `fig4_memory_effect.png`；右 seed_43 獎勵片段＋`ll.gif`（貼地懸停 agent）。
**口白：**
> 第二個結果，也是我覺得對在座最有意思的——記憶在密集獎勵不只沒用，是有害。
> 回到 LunarLander，比 B3-full 跟 B3-no-memory，這兩個唯一差別就是有沒有記憶。［指圖］no-memory 拿到 248 分、過了解題門檻；full 反而掉到 153，差 38%，p=0.0317，顯著。而三個稀疏環境的記憶效應都是小幅正向、不顯著——所以這個負向是密集環境獨有的。
> 我們挖了 LunarLander 五個 seed，其中 seed_43 只有 11.6 分。去看它第五輪 Gemma 寫的獎勵：一個很重的「接近地面就罰垂直速度」，加上一個「腳碰地就持續加分」。這兩項合起來，agent 學到的是貼著地面懸停、讓腳輕輕磨地——去拿那個持續的小獎勵，而不去完成最後落地、拿大的 terminal bonus。
> 背後的直覺，我們接 Ng et al. 1999 的 potential-based shaping：只有寫成勢能差的塑形，理論上才保證不改變最優策略。記憶讓 LLM 一輪一輪疊加塑形項，會越疊越偏離這個子類；原生訊號越強，這個漂移的代價就越明顯。這個機制我們沒有形式化證明，seed_43 是個案，但方向一致。

**數據直覺：** 原生獎勵已經夠好時，記憶逼 LLM 一直疊塑形 → 漂離 potential-based 子類 → 推離最優策略。

---

### 投影片 8｜結果三：變異性指紋（110 秒）
**畫面：** 左 `fig3_variance_signature.png`（四環境 std）；右 `fig5_per_iter_trajectories.png`（MC 單調 vs LL 震盪）。
**口白：**
> 第三個，變異性指紋，這個 pattern 很漂亮。
> ［指圖］看 B3-full 五個 seed 的標準差。MountainCar 是 3.08、Acrobot 4.39——五個 seed 幾乎收斂到同一個值，超穩。但 CartPole 是 113、LunarLander 是 91，差了一個數量級。
> 有趣的是，分界線不是獎勵密度，是「塑形空間有多豐富」。MountainCar、Acrobot 物理簡單、就一個子目標——蓄能、把桿擺上去——Gemma 每次都收斂到差不多那個近最優塑形，抽樣沒空間亂跑。CartPole 跟 LunarLander 塑形空間大，可以寫出很多「看起來都合理、但彼此衝突」的獎勵。
> ［指右圖］這張是單一 seed 跨五輪的軌跡。MountainCar 單調往上爬；LunarLander 上下震盪。同一套迭代機制，豐富塑形空間下就是混沌。
> 直覺：**塑形空間越大，LLM 寫得出的「似是而非」就越多，變異就越大。**

---

### 投影片 9｜結果四：模型無關（Part 2，70 秒）
**畫面：** `fig7_part2_hermes_vs_b0.png`，3 變體 × 4 環境格狀。
**口白：**
> 第四個，Part 2。我們問這個結論換 DQN 變體還成不成立。固定獎勵管線，只換 agent：vanilla、Double、Dueling。
> ［指圖］三個稀疏環境，九個格子全部正向；MountainCar 三個變體都顯著。密集的 LunarLander，三個變體全部負向。Part 1 的密度型態完整重現。
> 而且 Hermes 在三種變體之間互比，沒有任何一組顯著，p 全大於 0.3。意思是獎勵設計的效果跟你用哪種 value network **正交**——這套框架可以直接插拔到不同 DQN 上，不用重調。

---

### 投影片 10｜限制（70 秒）— **主動自曝**
**畫面：** 四條 bullet，圖示化（樣本數 / budget / B1 / 單一 LLM）。
**口白：**
> 限制我自己先講，免得等一下被問。
> 一，n=5，只有大效應（Cohen's d 大於 1）測得出來，中等效應在這個樣本數會 inconclusive。
> 二，訓練 budget 故意壓短，好讓 120 次跑得完，所以我們的 baseline 看起來比文獻弱——CartPole 的 B0 只有 154——這代表我們測的比較像「budget 內的收斂速度」，不是最終天花板。
> 三，B1 手工獎勵是我們自己寫的，只能說明「塑形在這個環境可行」，不能當真正的人類專家 baseline，所以我們的主張只靠 B0、B3、B3-no-memory。
> 四，只測了 Gemma 一個模型。另外 reward density 跟 shaping richness 在這四個環境是部分混淆的，要拆開得再加像 LunarLanderContinuous 這種環境。

---

### 投影片 11｜結論 + 可證偽預測（70 秒）
**畫面：** 上「reward density 是候選 predictor」「Memory: opt-in by task」；下三條 falsifiable predictions。
**口白：**
> 收尾。
> 這篇的一句話：EUREKA 那套 LLM 寫獎勵，搬到開源 Gemma，在稀疏獎勵成立、在密集獎勵反轉；而且記憶不是免費升級，在密集獎勵會顯著有害。我們提出 **reward density** 當作判斷它有沒有用的候選指標。實務建議是——**記憶應該按任務 opt-in，不是預設打開**。
> 留三個可證偽的預測：一，原生獎勵「密但沒對齊」的任務，記憶反而會幫忙，因為它能迭代修正 one-shot 修不掉的偏差；二，光靠 reward density 不夠，最後可能要「稀疏/密集 × 塑形空間豐富/貧乏」的二維分類；三，換更大的 LLM，變異會更誇張，因為它塑形更激進，上限更高、崩潰風險也更高。

---

### 投影片 12｜致謝 / Q&A（10 秒）
**畫面：** Thank You ＋ GitHub repo QR code。
**口白：**
> 以上，謝謝老師，這邊開放提問。

---

## Q&A 防禦準備（專家最可能問的 10 題）

| # | 老師會問 | 怎麼回 |
|---|---|---|
| 1 | n=5 太小，結論可靠？ | 用 Mann-Whitney + bootstrap，只宣稱大效應；負向結果 p=0.0317 能在 n=5 過，本身就代表是大效應；n=10/20 是 next step。 |
| 2 | 你的 baseline 太弱（CartPole B0=154，文獻能到 500）| 故意短 budget 讓 120 runs 可行；測的是「budget 內收斂速度」非天花板；長 budget replication 已列為 limitation。 |
| 3 | 記憶為什麼有害？機制證明了嗎？ | 沒形式化。Working hypothesis 是 potential-based shaping drift（Ng 1999）；seed_43 個案佐證；未來可加 formal verifier 把獎勵投影到勢能子類、量化漂移。 |
| 4 | 是 reward density 還是 shaping richness？兩者混淆 | 對，已列 limitation。CartPole(稀疏)、LunarLander(密集) 都 rich-shaping 卻同樣高變異；要拆得用 LunarLanderContinuous（密集但物理單純）。 |
| 5 | 為什麼只用 Gemma？換 GPT-4 會不一樣？ | 開源是 motivation（可重現/離線/成本）；變異指紋可能跟 Gemma 抽樣特性有關，Llama/Qwen replication 是 future work；我們預測更大模型變異更大。 |
| 6 | B1 你自己寫的，憑什麼說贏人類？ | 不宣稱。B1 只證明「塑形可行」；headline 只靠 B0 / B3 / B3-no-memory。 |
| 7 | 只跑 5 輪，更多輪會不會逆轉密集的負向？ | 沒測，已列 limitation；正是 falsifiable prediction 1 的內容。 |
| 8 | AST buffer 到底貢獻多少？ | B3-no-AST 在兩個 momentum 稀疏任務跟 full 在噪音內；差異集中在高塑形空間的 CartPole——buffer 的貢獻主要在 rich-shaping。 |
| 9 | 訓練用 shaped reward，評估怎麼比才公平？ | 評估一律用 env-native reward + 100 個 disjoint unseen seed（10000–10099），是唯一可跨條件比較的指標。 |
| 10 | 既然知道 potential-based，何不直接限制 Gemma 只能寫勢能差？ | 正是 future work（論文 5.3 第 2 點）；目前不約束，才看得到漂移代價——這是觀測，不是 bug。 |

---

## 製作備註（圖檔對應）

| 投影片 | 用哪個檔 | 位置 |
|---|---|---|
| 1 封面 | `grid_2x2.gif` | `paper/gifs/presentation/` |
| 2 鉤子 / 6 反轉 | `fig2_headline.png` | `paper/figures/` |
| 4 架構 | `fig1_architecture.png` | `paper/figures/` |
| 7 記憶有害 | `fig4_memory_effect.png` + `ll.gif` | `paper/figures/` + `paper/gifs/presentation/` |
| 8 變異 | `fig3_variance_signature.png` + `fig5_per_iter_trajectories.png` | `paper/figures/` |
| 9 Part 2 | `fig7_part2_hermes_vs_b0.png` | `paper/figures/` |
| 備用（Q&A） | `fig6_cartpole_boxplot.png`、`fig8_part2_hermes_robustness.png`、`dqn_compare_cp.gif`、Table 1 全表 | — |

**節奏提醒：** 投影片 2、7 是全場重心，講慢一點、留停頓。其餘結果頁維持「指圖→講數字→講直覺」三拍節奏，不要念表格。
