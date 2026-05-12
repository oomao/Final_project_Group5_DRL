## Why

`hermes-memory-layer` 提供 L2 sandbox(`multiprocessing.Process` 子程序驗證 + hard-kill)。對於課程 final 與單一 dev 機器,L2 已足夠擋住所有非對抗式 LLM 程式碼錯誤(syntax / infinite loop / OOM)。

但 L2 仍有兩個遺漏面:
1. **訓練時的 reward_fn 跑在主程序內**(L2 設計上的折衷,IPC 太貴)── 若 LLM 生成的程式碼在訓練第 800 ep 才表現出異常記憶體配置或 syscall,主程序仍會受影響
2. **檔案系統與網路完全敞開** ── reward_fn 雖然不能 import,但 `numpy.memmap` / `numpy.load` 之類能透過已暴露的 `np` 物件接觸主機檔案系統(我們 sandbox 沒堵)

本 change(**L3 容器級隔離**)透過 Docker container + nvidia-docker GPU passthrough,讓整個 `train.py` 跑在受限環境內:
- 唯一可寫目錄是 mounted `runs/` volume
- 網路預設關閉,Gemma API 透過 host-bind 反向代理(將呼叫限制到 generativelanguage.googleapis.com 一個 host)
- GPU 透過 `--gpus all` passthrough 但 driver/kernel 在 host

**這份 change 預期不會在課程 final demo 前 apply** ── 它的價值是:
- 寫進論文 Limitation / Future Work 段,展示我們意識到 L2 ≠ 容器級保障
- 給未來想把 Hermes-DQN 包成 SaaS / 跑使用者上傳 reward 的人留下實作藍圖
- 為 archive 後的歷史紀錄留下「我們的安全模型」明確說明

**Apply 的觸發條件**(此 change 暫不執行):
- 課程指導老師明確要求「能跑使用者上傳的 reward 程式碼」
- 進入 Phase 2 將 Hermes-DQN 推上 Hugging Face Space / 公開 demo
- 多人團隊共用同一台 4090 機器並開放 LLM-generated reward 自由實驗

## What Changes

- 新增 `Dockerfile` 在 repo 根目錄:
  - 基底 `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
  - 安裝 Python 3.11 + 從 `pyproject.toml` 同步相依
  - WORKDIR `/app`,COPY 必要程式碼(`hermes_dqn/`、`tools/`)
  - non-root user `hermes`(UID 1001)
- 新增 `docker-compose.yml`:
  - service `train`,build context 為 repo root
  - volume mounts:`./runs:/app/runs` 與 `./.env:/app/.env:ro`(readonly)
  - `gpus: all`(透過 nvidia-container-toolkit)
  - `network_mode: bridge` + egress allowlist 限制到 Google AI Studio API 端點
- 新增 `tools/run-in-docker.sh`(Linux/macOS)與 `tools/run-in-docker.ps1`(Windows + WSL2)兩個 wrapper,讓「`train.py --reward-source llm`」自動走容器
- 新增 `docs/sandbox-architecture.md` 描述 L1/L2/L3 各層責任與威脅模型
- (可選)新增 `Dockerfile.dev` for hot-reload 開發用(不啟用 production hardening)

## Capabilities

### New Capabilities

- `container-sandbox`:Docker + nvidia-docker 容器化 train.py 執行環境,提供檔案系統與網路層級隔離,GPU 透過 driver passthrough 維持訓練效能

### Modified Capabilities

- none(本 change 不修改既有 spec,只新增 container-sandbox capability;hermes-memory-layer 的 L2 sandbox 仍然存在於 image 內部作為最內層防線)

## Impact

- 新增檔案:`Dockerfile`、`docker-compose.yml`、`tools/run-in-docker.{sh,ps1}`、`docs/sandbox-architecture.md`、(可選)`Dockerfile.dev`
- 修改檔案:`README.md`(新增「Container mode」段落)、`hermes_dqn/README.md`(相同)、`.dockerignore`
- 新增相依(系統層,非 Python):
  - Docker Engine 24+(host)
  - nvidia-container-toolkit(host,for GPU passthrough)
  - Windows 上:Docker Desktop + WSL2 + NVIDIA driver ≥ 525
- 不修改任何 Python 程式碼 ── L3 是「**在外面包一層**」,不改裡面
- Apply 觸發條件:見 proposal Why 段最後一段
- 引用的 spec:
  - `establish-project-lifecycle-spec / env-setup`:Requirement "Python Version Lock" 與 "CUDA and PyTorch Wheel Lock"(container image 內必須符合)
  - `hermes-memory-layer / reward-sandbox`(L2,容器內仍存在)
  - `gemma-reward-generator / llm-reward-client` Requirement "API Key Management via .env"(讀掛載的 `.env`)
