## Why

Hermes-DQN 目前只完成 `bootstrap-dqn-baseline`,後續仍有 4 個重大 change(`gemma-reward-generator`、`hermes-memory-layer`、`ast-buffer-manager`、`closed-loop-fitness`)要在期末前由 4 人小組完成,並提交論文、簡報、demo 影片做口試。在這個十字路口,專案最容易出的問題不是技術,而是「規範漂移」:

- 文件:OpenSpec 四件套是否每個 change 都齊?白話介紹要不要跟著更新?註解用中文還英文?
- 環境:同學 A 用 pip + venv,同學 B 用 conda,同學 C 上 4090 跑出來的結果別人重現不了。
- 實驗:每個條件跑幾個 seed?何時可以 early-stop?run 目錄怎麼擺?
- 評估:哪些 baseline 一定要比?p 值多少算贏?outlier 要不要拿掉?
- 交付:論文格式、簡報張數、影片長度、每個人各負責哪一塊?

這個 change 本身**不寫任何程式碼**,而是把這 5 件事各自形成一份規範 spec,並把目前 3 位討論代理人(文件規範/環境設置/實際實驗/評估結果/文件完成)提出的主張收斂成單一決議。後續每一個 feature change 都必須在 proposal 內引用本 change 產出的 capability spec 中對應的 scenario,確保 4 個成員、5 個 change、6 個 baseline 條件、最後 1 場口試之間沒有口頭協議。

## What Changes

- 新增 5 個 capability spec,作為所有後續 change 的治理基準:
  - `doc-standards` — OpenSpec 四件套規則、handover 節律、註解語言、白話介紹同步、PR 驗證閘門
  - `env-setup` — Python 3.11、cu121、uv 為主 (pip+venv fallback)、.env 規則、pre-commit、smoke test、4090 共享協議
  - `experiments-protocol` — 5 seed/condition、1500 ep 固定預算、Hermes 外層 5 次 LLM iter、三層 run 目錄、reward_fn.py + SHA-256
  - `evaluation-criteria` — 6 baseline、Mann-Whitney U + bootstrap CI、Win 三條件、primary metric = converge_episode、可重現七件套
  - `final-deliverables` — IEEE 雙欄 8 章論文、18-22 張 5 段簡報、4-6 分鐘 demo 影片、第 N 版同步、4 人分工
- 把 3 個討論代理人之間的衝突收斂成單一答案,理由寫在 `design.md`:
  - 環境管理工具 → 採用 **uv**,pip + venv 留作備援,並在 env-setup spec 寫清楚切換指令
  - run/ 目錄階層 → ad-hoc 訓練保留 timestamp,正式評估強制 `runs/<exp>/<condition>/seed_NN/`
  - 程式碼註解語言 → **英文 only**(WHY-only 原則)
  - 4090 佇列 → GitHub Issues 預約 + 啟動時 `nvidia-smi` 守門,兩層並行

## Capabilities

### New Capabilities
- `doc-standards`: OpenSpec 文件、handover、註解、白話介紹、驗證閘門的治理規範
- `env-setup`: Python 環境、相依鎖定、API 金鑰、可重現性邊界、4090 共享協議
- `experiments-protocol`: 訓練條件矩陣、seed 政策、run artifact 規格、Hermes 外層迴圈規則
- `evaluation-criteria`: Baseline 集合、統計檢定、Win 判定、報告欄位、可重現七件套
- `final-deliverables`: 論文、簡報、demo 影片、版本號、4 人分工的交付規格

### Modified Capabilities
- none(現有 `dqn-baseline`、`reward-plugin`、`fitness-evaluation`、`numbering-rule` 不受影響)

## Impact

- 本 change 本身**不引入任何程式碼**,純粹是治理 spec。
- 引用方:接下來的 4 個 feature change 在 `proposal.md` 必須引用本 change 對應 scenario。
- 後續實作影響(由各自的 change 完成,不在此 change 範圍):
  - `requirements.txt` → `pyproject.toml` + `uv.lock` 由 env-setup 的後續 change 完成遷移
  - 新增檔案:`.github/PULL_REQUEST_TEMPLATE.md`、`.pre-commit-config.yaml`、`.env.example`、`.vscode/extensions.json`、`tools/cleanup_runs.py`、`scripts/claim-gpu.sh`、`docs/4090-booking.md`
  - run 目錄階層由 `experiments-protocol` 引用 change 統一遷移為三層
- 沒有破壞性變更:`bootstrap-dqn-baseline` 的成品(flat timestamp run、`requirements.txt`)在過渡期仍合法。
