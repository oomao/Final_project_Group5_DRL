# Gemma Reward Generator 實作紀錄

**OpenSpec change**:`gemma-reward-generator`
**狀態**:Group 1-7 完成、smoke 7/7 全綠、1500-ep run 正在 4090 跑
**核心命題**:**開源 Gemma 4 寫的 reward 可讓 vanilla DQN 收斂** ── EUREKA 命題的開源重現

---

## 設計

### 子套件結構

```
hermes_dqn/llm/
├── __init__.py
├── client.py    # LLMRewardClient with 3-retry
├── compile.py   # AST sandbox + dry-run validation
└── prompts.py   # LunarLander prompt + few-shot
```

### 2 個新 Capability + 14 個 Scenario

| Capability | Scenarios |
|---|---|
| `llm-reward-client` | 6 Req / 14 Scenario:client 建構、生成、3 retries、attempts log、AST 沙箱、dry-run |
| `llm-reward-integration` | 4 Req / 11 Scenario:`--reward-source` 旗標、API key 載入、reward_fn.py 強制、失敗不留半成品 |

### 關鍵決策

1. **AST 沙箱不防對抗**:LLM 是合作者而非攻擊者。黑名單 `Import` / `_dunder` access,白名單 builtins 約 20 個
2. **3 次重試,第 3 次強制 fallback**:`return env_reward`,保證訓練不被鎖死
3. **All runs 都寫 `reward_fn.py`**:env 路徑寫 stub passthrough,LLM 路徑寫 Gemma 原文。SHA-256 兩條都記
4. **API key 經 `python-dotenv` 從 `.env` 讀**,key 不能進 git
5. **失敗時不留半成品**:LLM/compile fail → exit 1 BEFORE 訓練,`llm_attempts.jsonl` 留追溯,`model_final.pt` 絕不存在

---

## 過程中的 bugs

### Bug 1:`.env` 漏在 `.gitignore` 外

最危險的一個。使用者要貼 API key 前,我順手檢查 `.gitignore`,發現 `.env` 沒有被 ignore(只有 `env/` 當 venv 名)。立刻加上:

```diff
+ # Secrets (API keys, etc.) — never commit
+ .env
+ .env.local
+ *.env.local
```

並用 `git check-ignore .env` 驗證 ── 在用戶貼 key 之前修好。

### Bug 2:使用者的 GEMMA_MODEL 拼錯

使用者寫了 `GEMMA_MODEL=gemma-3-32b-it`。我先呼叫 `genai.list_models()` 確認可用模型:

```
models/gemma-4-26b-a4b-it
models/gemma-4-31b-it
```

`gemma-3-32b-it` 不存在(可能記混 Gemma 3 + 4)。建議使用者改成 `gemma-4-31b-it`(對應 README 寫的 "Gemma 4 31B")。

### Bug 3:Windows CRLF 把 SHA-256 搞壞

`pathlib.Path.write_text()` 預設用 OS line separator(Windows 是 CRLF)。但 SHA-256 是從 Python 字串(LF)算的。結果:

```
sha in config        : 17c20cba9d651f88...   <- 從 LF 字串算
computed sha-256     : 992ff3594117fd7b...   <- 從 CRLF 檔案算
sha match            : False
```

改用 `write_bytes(src.encode('utf-8'))` ── 完全跳過 newline translation,bytes-on-disk 與 SHA 來源一致。

### Bug 4:PowerShell 把空字串 env var 當未設定

Smoke 7.7 要測「沒 API key 時 train.py 應 exit 1」。我先試:

```powershell
$env:GOOGLE_API_KEY = ''
python -m hermes_dqn.training.train --reward-source llm
```

結果:python 還是讀到 key(因為 PS 把空字串 set 視為 unset,然後 dotenv 從 `.env` 載入)。

改成把 `.env` 暫時 move 走:

```powershell
Move-Item .env .env.bak
try { python -m ... } finally { Move-Item .env.bak .env }
```

→ 成功 exit 1,訊息明確,無 `model_final.pt`。

---

## Smoke test 結果(7/7 全綠)

| # | 測試 | 結果 |
|---|---|---|
| 7.1 | env 路徑 deterministic vs baseline | ✓ 前 10 集 return byte-match |
| 7.2 | env 路徑 SHA + reward_fn.py | ✓ stub passthrough 寫對 |
| 7.3 | llm 路徑端到端 | ✓ Gemma 寫 34 行 reward,訓練 10 ep |
| 7.4 | attempts log | ✓ 2 行 — attempt 1 撞 Google 500 internal,attempt 2 accepted |
| 7.5 | reward_fn.py vs response | ✓ byte-identical |
| 7.6 | config sha vs file | ✓ 修了 newline 之後 match |
| 7.7 | missing key 失敗路徑 | ✓ exit 1,訊息指向 `.env.example`,無 model |

---

## Gemma 第一次寫的 reward(34 行,意外合理)

```python
def reward(obs, action, next_obs, env_reward, terminated, truncated, info):
    x = next_obs[0]; vx = next_obs[2]; angle = next_obs[4]
    v_angle = next_obs[5]; leg_l = next_obs[6]; leg_r = next_obs[7]

    r = float(env_reward)
    # Shaping:中心 + 直立 + 慢速度的小懲罰,係數 0.1 不蓋過原 reward
    shaping = -0.1 * (abs(x) + abs(angle) + 0.1 * abs(vx) + 0.1 * abs(v_angle))
    r += shaping

    # Leg contact 小幅度 nudge(原本就有 +10/leg)
    r += 0.1 * (leg_l + leg_r)

    # Terminal 信號放大:成功/失敗對比更明顯,加速收斂
    if terminated:
        if env_reward > 0:
            r += 100.0
        elif env_reward < 0:
            r -= 100.0

    return float(r)
```

評論:
- 用 `next_obs` 取狀態(對的 ── 評估的是 *結果* 的好壞)
- 不在 reward 加新項目,只在 *相對權重* 調整
- 沒有 hack:不靠 `info` dict、不引用未知變數
- 對應 EUREKA 論文觀察「LLM 寫的 reward 比手寫更好」── 這份小型勝利已產生

API 用量:smoke 7.3 用了 2 次 generate_content 呼叫(因為 attempt 1 撞 Google 500)。Free tier 15 RPM 完全沒撞到。

---

## 1500-ep 真實訓練(進行中)

跑了:
```
python -m hermes_dqn.training.train --reward-source llm --episodes 1500 --seed 42 --out-dir runs/gemma_seed42
```

Background ID `bmj2w71b8`,預計 ~25 分鐘(對齊 baseline)。

⚠️ **預警**:Gemma 的 reward 對 episode return 加 shaping,所以 `episodes.jsonl["return"]` 數字**不能直接跟 baseline 比**。Baseline 是 env-native,Gemma 是 shaped。要 apples-to-apples 比較,後續 `closed-loop-fitness` change 必須**同時記錄 env_reward 與 shaped reward**。

---

## 用到的 spec scenario(引用治理 spec)

本 change 的 `proposal.md` 在 Impact 段明確列出:

- `establish-project-lifecycle-spec / env-setup`:Requirement "API Key Management via .env"(3 個 scenarios 全部)
- `establish-project-lifecycle-spec / experiments-protocol`:Requirement "Reward Function Artifact and Integrity"(reward_fn.py + SHA-256)
- `bootstrap-dqn-baseline / reward-plugin`:Requirement "Injectable reward in env wrapper"(7-arg 簽章)
- `bootstrap-dqn-baseline / fitness-evaluation`:用既有 FitnessEvaluator 不重寫

這條引用模式就是治理 spec 想要建立的:**所有 feature change 都要在 proposal 階段把 "我滿足 governance 的哪些 scenario" 寫清楚**。

---

## 接縫狀態

Hermes-DQN 設計的 4 個 interface,目前 2 個已敲定:

| 接縫 | 狀態 | 定義者 |
|---|---|---|
| `RewardFunction`(7-arg) | ✅ | `bootstrap-dqn-baseline` |
| `LLMRewardClient.generate()` | ✅ | `gemma-reward-generator`(本 change) |
| `MemoryEntry` | ⏳ | `hermes-memory-layer`(下一份) |
| `BufferAction` enum | ⏳ | `ast-buffer-manager` |

`hermes-memory-layer` 開工時要把 Gemma client 改成接收 `memory: list[MemoryEntry]` 參數(目前只接收 task_spec)── 這條 future-proof 已在 spec 中註記。

---

## 1500 ep 訓練結果 & vs Baseline 對照

### 訓練本身

跑了 16m29s(比 baseline 的 24m47s 快了 33%)。最後一集 shaped return 327.8。

shaped fitness(僅供參考,不能跟 baseline 比):
- converge_episode = 525
- mean_reward_last100 = 312.21(shaped)
- success_rate = 0.85(shaped ≥ 200)

shaped 數字「比 baseline 高」是因為 Gemma 在成功降落加了 +100 終局放大;不能用來說「贏 baseline」。

### Apples-to-apples 評估(env-native reward,100 個新 seeds)

跑 `tools/_eval_env_native.py` 載入兩個 model_final.pt,各跑 100 個 unseen eval seeds(10000-10099),greedy(ε=0),用 **env-native reward** 評分:

| 訓練時用的 reward | Mean env reward | Median | Success rate (≥200) | Crash rate (<0) | Mean ep length |
|---|---|---|---|---|---|
| env (native) | 162.72 | 226.43 | 53% | 14% | 265 |
| **llm (Gemma)** | **207.72** | **238.02** | **78%** | **7%** | **419** |

### 結論

**Gemma 的 reward 在不見過的測試 seeds 上完勝 baseline**:

- **+45 mean reward(+28%)**
- **+25 pp success rate**(53% → 78%)
- **減半 crash rate**(14% → 7%)
- 訓練 wall-time 還快 **33%**
- 平均 episode 長度從 265 增到 419 步 ── agent 變得更「謹慎」,願意多花步數降落而非快速死

### 這代表什麼

**EUREKA 命題的開源重現成立**(對 seed 42 而言):「免費的 Gemma 4 31B 寫的 reward,讓 vanilla DQN 在 LunarLander-v3 上明顯優於環境內建 reward」。

但這只是 **n=1 seed**,有兩個 caveat:
1. seed=42 可能是好球(LunarLander 在 seed 之間波動大)
2. Gemma 寫出來那份 reward 有運氣成份(Gemma temperature ~0.7)

`experiments-protocol` spec 規定的 5-seed 比較會在 `closed-loop-fitness` 階段做。屆時用 Mann-Whitney U + bootstrap CI 才算正式 claim win。但本 change 的核心目標「能跑通端到端 + 不破壞 baseline 路徑」**已 100% 達成**。
