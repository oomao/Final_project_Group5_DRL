# Claude 實作起步 — Session 總覽

**日期**:2026-05-12
**AI**:Claude Opus 4.7 (1M context)
**人員**:csm088220@gmail.com (1 名,本 session 為個人推進)
**Session 範圍**:從零實作整套 Hermes-DQN ── README 三大核心貢獻全部完成 + L2 sandbox + 7 步閉環 + 統計工具 + 7 個 change archive,可跑可重現

---

## Session 起點 vs 終點

| 面向 | Session 開始 | Session 結束 |
|---|---|---|
| **Python 程式碼** | 完全沒有 | 完整 `hermes_dqn/` package(7 個子模組,~5000 行) |
| **DQN baseline** | 設計只在 README | 訓練可跑通,seed=42 收斂於 ep 399、success 95% |
| **LLM reward 通道** | 概念 | 完整 Gemma 4 31B 接入 + L2 子程序 sandbox + 3-retry,單發勝 baseline +28% |
| **記憶系統** | 概念 | SQLite FTS5 長期記憶層運作,跨 iter 累積 priors 已驗證 |
| **AST/Buffer 管理** | 概念 | 純函式庫 + ReplayBuffer save/load + 4 種 diff 分類 + 3 種 buffer 政策 |
| **閉環引擎** | 不存在 | 7 步閉環 in-process 多輪迭代,pilot 3 iter × seed 42 驗證 |
| **統計工具** | 不存在 | Mann-Whitney + bootstrap CI + 三條件 Win 判定 + 報告生成 |
| **治理規範** | 散落 / 未成文 | 46 個 SHALL/MUST Requirement,5 個 capability spec |
| **OpenSpec changes** | 2(`improve-dev-scripts` + `numbering-rule`) | **9**(7 archived + 1 active + 原本的 numbering-rule) |
| **永久 capability spec** | 1(numbering-rule) | **19**(全部位於 `openspec/specs/`) |
| **可視化** | 無 | pygame `play.py` + 訓練曲線動畫 + 遊戲對打 GIF |
| **安全** | `.env` 未在 `.gitignore`(漏洞) | `.env` + `.env.local` 都在 ignore,API key 保護到位 |

---

## 三大成果一句話描述(對應 README 三大核心貢獻,**全部到位**)

1. **開源化(Gemma)**:Google Gemma 4 31B → 7-arg Python reward → AST 沙箱 + L2 子程序隔離 + 3-retry → DQN 訓練。Gemma 單發勝 baseline +28% env_native_mean(207.72 vs 162.72)。EUREKA 命題開源重現成立(seed=42)
2. **記憶擴增**:SQLite FTS5 長期記憶層,跨 iter 把過去 reward + fitness 當 prior 餵給 Gemma。Pilot 確認 iter 2/3 prompt 真的含 PRIOR HIGH-FITNESS ATTEMPTS
3. **AST 感知緩衝區**:`diff_rewards` 4 種分類 + KEEP/DECAY/CLEAR 政策 + ReplayBuffer save/load。Pilot 觀察到 sim=0.71 → DECAY、sim=0.56 → CLEAR,完全符合設計

**加碼(README 未列但實作了)**:7 步閉環引擎、Mann-Whitney + bootstrap 統計工具、治理 spec 46 條(5 個 governance capability)、L2 子程序 sandbox。

---

## 詳細紀錄索引

| 檔案 | 主題 |
|---|---|
| [02-baseline實作.md](02-baseline實作.md) | DQN baseline change 的完整過程(skeleton → 收斂結果) |
| [03-白話架構文件.md](03-白話架構文件.md) | `白話架構介紹.md` 的設計理念(籃球教練比喻) |
| [04-治理spec多agent協作.md](04-治理spec多agent協作.md) | 3+1 agent 模式建立 `establish-project-lifecycle-spec` |
| [05-Gemma實作.md](05-Gemma實作.md) | `gemma-reward-generator` change + Gemma 第一次寫的 reward |
| [06-記憶與sandbox.md](06-記憶與sandbox.md) | `hermes-memory-layer` ── SQLite FTS5 長期記憶 + L2 子程序 sandbox |
| [07-AST緩衝區.md](07-AST緩衝區.md) | `ast-buffer-manager` ── AST diff + buffer 政策純函式庫 |
| [08-閉環整合.md](08-閉環整合.md) | `closed-loop-fitness` ── 7 步閉環引擎 + Mann-Whitney 統計工具 + pilot 結果 |
| [09-archive與收尾.md](09-archive與收尾.md) | 7 個 change archive、19 個永久 capability、session 全景 |

---

## 重要決策摘要(寫進治理 spec 的)

| 決策面 | 結論 | 理由 |
|---|---|---|
| 環境管理 | **uv 為主、pip+venv 備援** | 安裝快 10-100×,跨平台一致 |
| Python 版本 | `>=3.11,<3.12` | 3.12 改了 ast 模組,會影響 ast-buffer-manager |
| CUDA wheel | `torch==2.5.1+cu121` 鎖定 | 4090 在 cu121 已驗證 |
| 註解語言 | **英文 only, WHY-only** | 程式碼 identifier 是英文,混排不一致 |
| 中文用途 | README、proposal、handover、白話介紹 | 受眾是台灣團隊 + 教師 |
| Seed 數 | 每 condition **5 seeds [42-46]** | n=5 + Mann-Whitney 平衡 power 與 4090 throughput |
| 訓練預算 | **固定 1500 ep,禁用 early-stop** | 公平的 converge_episode 比較 |
| Hermes 外層迴圈 | **5 次 LLM 重寫** | EUREKA 觀察:3 次太少,10 次爆 compute |
| 統計檢定 | **Mann-Whitney U + 5000 bootstrap CI** | n=5 不能假設常態 |
| Win 條件 | **p<0.05 AND ≥10% 差 AND CI 不重疊** | 三條件齊全才算贏,防 cherry-picking |
| Primary metric | **converge_episode**(sample efficiency) | 本專案賣點是「LLM reward 讓 DQN 學更快」 |
| Outlier 處理 | **全部 report,不剔除** | 學術誠實 + DRL 本來就 noisy |
| 4090 共享 | **GitHub Issues 預約 + nvidia-smi 守門** | 雙層協議 |
| Reward 沙箱 | **AST 黑名單 + builtins 白名單**,不防對抗 | LLM 是合作者非攻擊者 |

---

## 主要技術 caveat / 待解問題

1. **n=1 太吵,無法 claim memory 有效**:閉環 pilot 結果 181 → 90 → 168(非單調)。唯一單調是 crash rate(22% → 12% → 2%)。需 5 seed + Mann-Whitney 才能下結論 ── 留給實驗週
2. **訓練 reward 跑在主程序內(L2 sandbox 是驗證隔離,不是訓練隔離)**:LLM 寫的 reward 第 800 ep 才出 bug 仍會影響主程序。L3 容器化(`reward-sandbox-isolation`)是 proposal-only,觸發條件式
3. **Gemma temperature 未固定**:預設 ~0.7,同 seed 不同次跑 reward 不同。可重現性七件套已收錄 prompt+response log,但跨 run 結果仍有差異 ── 統計檢定時 sample size 要夠
4. **CI 尚未建立**:目前靠 pre-push hook + reviewer 人工
5. **uv 遷移未做**:治理 spec 規定 uv 為主,但 pyproject + requirements.txt 仍是 pip 風格
6. **B1 hand-shaped reward 沒寫**:治理 spec 規定要非作者第三人寫,等組員協調

---

## 後續路線圖(本 session 完成狀態)

1. ✅ `bootstrap-dqn-baseline` ── archived(2026-05-12-bootstrap-dqn-baseline)
2. ✅ `establish-project-lifecycle-spec` ── archived(2026-05-12-establish-project-lifecycle-spec)
3. ✅ `gemma-reward-generator` ── archived(2026-05-12-gemma-reward-generator)
4. ✅ `hermes-memory-layer` ── archived(2026-05-12-hermes-memory-layer),含 L2 sandbox
5. ✅ `ast-buffer-manager` ── archived(2026-05-12-ast-buffer-manager)
6. ✅ `closed-loop-fitness` ── archived(2026-05-12-closed-loop-fitness),pilot 驗證通過
7. 📝 `reward-sandbox-isolation` ── proposal-only(觸發條件式)

每一份 change 結束時都是**獨立可發表的命題**:
- ② Gemma 證明「LLM 寫的 reward 比 env-native 好」── ✅ EUREKA 開源重現(n=1)
- ③ 記憶證明「有記憶的 LLM 比沒記憶更快收斂」── 🟡 機制驗證,實驗統計留給實驗週
- ④ AST/Buffer 證明「換 reward 時保留有用經驗比清空好」── 🟡 機制驗證(pilot 觀察到正確分類)
- ⑤ 整合 + 統計 ── ✅ 工具與引擎就緒,等實驗週填數據

---

## Memory(我下次會自動記得的)

存在 `~/.claude/projects/.../memory/`:

- `project_team_context.md` ── 這是 4 人團隊作業,scoping 要考慮 ownership 與口試答辯責任
- `feedback_doc_style.md` ── 你要白話文件時的固定樣板:單一主比喻 + 術語翻譯表 + 口試版收尾 + 繁體中文
- `project_lifecycle_spec.md` ── `establish-project-lifecycle-spec` 已建立,46 個 Requirement,未來任何 change 都要引用

下次 session 開新 change 時,我會自動帶這些 context。
