# DQN Baseline 實作紀錄

**OpenSpec change**:`bootstrap-dqn-baseline`
**狀態**:全 37 個 task 完成,strict validate 通過
**收斂結果**:seed=42 第 399 ep 收斂、mean reward 262.8、success rate 95%、24m47s on RTX 4090

---

## 起點

- 整個 repo 沒有任何 Python 實作程式碼
- 只有架構文件、簡報、論文 v1/v2、OpenSpec workflow 腳手架
- 需要建立「能承載 LLM-generated reward 的 DQN 訓練底盤」

## 設計階段(opsx:propose)

### Capabilities(3 個)

| Capability | 對應檔案 | 核心 Requirement 摘要 |
|---|---|---|
| `dqn-baseline` | `hermes_dqn/agent/`、`hermes_dqn/training/` | 訓練可重現、CPU/GPU、target sync、ε 衰減、寫 JSONL + config + model_final |
| `reward-plugin` | `hermes_dqn/env/reward.py`、`lunar_lander.py` | 7-arg Protocol、env wrapper 注入、例外傳播 |
| `fitness-evaluation` | `hermes_dqn/training/fitness.py` | `FitnessReport(converge_episode, mean_reward_last100, success_rate, total_episodes, seed)` |

### 關鍵設計決策(寫在 design.md)

1. **Package 結構**:`hermes_dqn/{env,agent,training,utils}/` 子套件而非單一檔 ── 未來加 llm/、memory/、buffer/ 不用 churn-only rename
2. **`RewardFunction` Protocol 7 參數**:`(obs, action, next_obs, env_reward, terminated, truncated, info) -> float` ── 廣到讓 Hermes 寫的 reward 不用改簽章
3. **`FitnessEvaluator(success_threshold=200.0, window=100)`**:對齊 LunarLander 官方成功標準
4. **JSONL 不用 TensorBoard**:Hermes 後續需要 *programmatic* 訪問,UI 不適合
5. **Vanilla DQN(MLP 64-64, lr 5e-4)**:不做 Rainbow,瞄準 IJRPR 2025 baseline 數字

---

## 實作階段(opsx:apply, 9 個 task group)

### 順序

```
1. Skeleton (init.py + pyproject + requirements + gitignore + README)
2. Utils (set_global_seed)
3. Env (RewardFunction Protocol + RewardInjectionWrapper + make_env)
4. Agent (QNetwork + ReplayBuffer + DQNAgent + DQNConfig)
5. Training loop (JsonlLogger + train.py + argparse + main)
6. Fitness (FitnessReport + FitnessEvaluator)
7. Smoke test (10 ep + fitness eval + determinism check)
8. Baseline run (1500 ep + convergence + README update)
9. Wrap-up (strict validate + scenario verification)
```

### 過程中的 bugs

1. **Windows cp950 unicode 錯**:`print("✓ Run complete...")` 在 PowerShell 撞 cp950,改成 `[OK]`
2. **`weights_only` FutureWarning**:torch.load 預設要改為 `weights_only=True`,我們的 checkpoint 自家存的,加 flag 抑制警告
3. **Determinism**:`wall_time_s` 因 wall-clock 自然不一致,但 `return` 欄位 byte-identical ── spec 只要求後者,我寫了專用比對

---

## Smoke test 結果(Task 7.x)

| 測試 | 期望 | 實際 |
|---|---|---|
| 7.1 跑 10 ep | 產出 config + jsonl(10 行) + model | ✓ |
| 7.2 fitness eval | total=10, converge=None, success=0, seed=42 | ✓(全符合) |
| 7.3 determinism | 同 seed 兩次跑 return 一致 | ✓ byte-identical |

---

## Baseline 收斂結果(Task 8.x)

跑了 `python -m hermes_dqn.training.train --episodes 1500 --seed 42 --out-dir runs/baseline_seed42`

### Fitness 報告

```
FitnessReport(
  converge_episode    = 399       <- 比文獻 IJRPR 2025 ~1200 快 3 倍
  mean_reward_last100 = 262.79    <- 目標 ≥ 200 ✓
  success_rate        = 0.95      <- 目標 ≥ 0.90 ✓
  total_episodes      = 1500
  seed                = 42
)
Wall time: 24m47s on RTX 4090 / cu121
```

### 對 baseline 表現的解讀

兩個假說:
1. **Seed 運氣**:LunarLander 在不同 seed 之間波動大,seed=42 可能是好球
2. **Hyperparams 偏好**:lr=5e-4 + smooth_l1_loss + 1000-step hard target sync 可能比文獻更激進

不能用單 seed 下定論。`experiments-protocol` 治理 spec 規定:正式比較必須跑 5 個 seed [42-46]。屆時統計結果才是「baseline 真實水準」。

---

## 動畫(play.py)

跑完 baseline 之後,使用者想看 agent 實際飛。新增 `hermes_dqn/training/play.py`:

- `--run-dir runs/baseline_seed42 --episodes N` 載入 model + 開 pygame 視窗
- `--epsilon 0` 純貪婪 playback

3 集結果:
- Episode 1: return = 303.8 LANDED
- Episode 2: return = -30.3 crashed
- Episode 3: return = 292.3 LANDED
- 2/3 成功降落

成功率比訓練尾段的 95% 低是合理的 ── n=3 太少 + 用未訓練 seed(1000+)。

---

## Spec 引用關係(後續 change 會引用)

所有後續 change 都要引用本 change 的 spec:

- `gemma-reward-generator` 引用 `reward-plugin / Injectable reward in env wrapper`(7-arg 簽章)
- `closed-loop-fitness` 引用 `fitness-evaluation / FitnessEvaluator reads JSONL logs`
- 任何要跑訓練的 change 都引用 `dqn-baseline / Train DQN on LunarLander-v3 end-to-end`

---

## Files 產出

```
hermes_dqn/
├── __init__.py
├── README.md
├── env/
│   ├── __init__.py
│   ├── lunar_lander.py    # make_env + RewardInjectionWrapper
│   └── reward.py          # RewardFunction Protocol + default_reward_fn
├── agent/
│   ├── __init__.py
│   ├── q_network.py       # MLP 64-64
│   ├── replay_buffer.py   # numpy circular buffer
│   └── dqn_agent.py       # DQNAgent + DQNConfig
├── training/
│   ├── __init__.py
│   ├── train.py           # main entry point
│   ├── play.py            # pygame playback
│   ├── fitness.py         # FitnessReport + FitnessEvaluator
│   └── logger.py          # JsonlLogger
└── utils/
    ├── __init__.py
    └── seeding.py         # set_global_seed

pyproject.toml             # torch / gymnasium[box2d] / numpy / tqdm
requirements.txt
.gitignore
runs/                      # gitignored, training artifacts
└── baseline_seed42/
    ├── config.json
    ├── episodes.jsonl     # 1500 行
    └── model_final.pt
```

[hermes_dqn/README.md](../../hermes_dqn/README.md) 的 Baseline runs 表已加入第一筆紀錄。
