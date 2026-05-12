## Context

`hermes-memory-layer` 把 reward 編譯與 dry-run 移進 `multiprocessing.Process`(L2),擋住 LLM 寫的程式碼把主訓練程序 kill / OOM。但 L2 是「程序級」隔離,reward_fn 訓練時仍跑在主程序內,且整台主機檔案系統與網路對該 process 完全可見。

本 change(L3)用 Docker container 把**整個 `train.py` 程序**包起來。容器內部仍保留 L2 sandbox 作為深度防禦;容器邊界則處理「reward 透過 `np.memmap` 之類接觸主機 `/etc/passwd`」這類 L2 抓不到的情況。

威脅模型對應:

| 威脅 | L1 (AST/builtins) | L2 (subprocess) | L3 (container) |
|---|---|---|---|
| `import os` | ✅ | ✅ | ✅ |
| `open("...")` | ✅(builtin 白名單) | ✅ | ✅ |
| `while True: pass` | ⚠️ thread join soft | ✅ hard kill | ✅ |
| 巨量記憶體配置 | ❌ | ⚠️(只在 Linux 有 RLIMIT) | ✅(cgroup) |
| `np.memmap("/etc/passwd")` | ❌(np 是白名單) | ❌(主機檔案系統可見) | ✅(volume mount only) |
| 對外發送資料 | ❌ | ❌ | ✅(network allowlist) |
| 影響主機 GPU driver | ❌ | ❌ | ⚠️(driver passthrough,kernel exploits 仍存在) |

## Goals / Non-Goals

**Goals:**
- `Dockerfile` 能 build 出可跑完整 1500-ep 訓練的 image
- 容器內訓練吞吐量 ≥ 主機原生 90%(GPU passthrough 損耗 ≤ 10%)
- 可重現:同 `pyproject.toml` + 同 git commit SHA 在不同 host 建出相同 image(透過 BuildKit + 鎖定 base tag)
- 文件清楚:`docs/sandbox-architecture.md` 描述 L1/L2/L3 各自負責什麼威脅
- 對於 Windows host:給出 WSL2 + Docker Desktop 安裝步驟,並列出 known issues(nvidia-container-toolkit 在 Windows 上的歷史 bug)

**Non-Goals:**
- 加 Kubernetes / orchestration(本 change 只做單機 container)
- 加 seccomp profile / AppArmor(留給後續若上 production)
- 跑多容器並行訓練(`closed-loop-fitness` 階段再說)
- 修改 Python 程式碼(L3 是外殼,內部維持 L2 完整保護)
- 在課程 final demo 前 apply 這份 change(本 change 預期維持 proposal-only 狀態 ≥ 6 個月)

## Decisions

### A. 基底 image 選擇

`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`:
- 對齊 `env-setup` spec 規定的 cu121
- runtime tag(無 dev tools)降低 image 大小;若需要編譯 C extension 改 devel
- Ubuntu 22.04 LTS 是 nvidia-container-toolkit 文件覆蓋最完整的 host
- 大小約 3-4 GB(可接受)

**Alternative considered**:
- `python:3.11-slim` + 手動裝 CUDA → 被否決,nvidia 已提供官方 image
- `pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime` → 被否決,pytorch 官方 image 不接受 build-args(我們要 install --user 自己的 deps)
- Distroless → 被否決,缺 bash / curl 之類除錯工具,單機 demo 過度

### B. Non-root user

容器內預設用 `hermes` user(UID 1001)。`runs/` volume mount 需要對應 host 端 chown(在 `run-in-docker.sh` wrapper 內處理)。`.env` 用 `:ro` mount 防止容器寫回。

**Alternative considered**:跑 root → 被否決,即使容器內 root 也比 user 危險;對 demo 場景 chown 一次值得。

### C. 網路策略

預設容器走 `bridge` 網路,但 iptables / docker network rules 限制 egress 只允許:
- `generativelanguage.googleapis.com:443`(Gemma API)
- `pypi.org` / `download.pytorch.org`(build 階段)
- DNS

build 與 runtime 用不同 network config:
- build 階段:正常 bridge 網路抓 pip wheel
- runtime 階段:bridge + post-up iptables drop everything except allowlist

實作層面:
- 用 `docker-compose.yml` 的 `networks` + `extra_hosts` 控制
- 若要更嚴,改用 Cilium / Calico,但本 change 不上 K8s,bash iptables 就夠

**Alternative considered**:
- `network_mode: none` → 被否決,Gemma API 呼不到
- `network_mode: host` → 被否決,失去網路隔離意義
- VPN / outbound proxy → 被否決,setup 太重

### D. GPU passthrough

`docker-compose.yml` 用:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```
(`gpus: all` 是舊語法,新版 compose 用上面這個)

要求:
- Host 端裝 nvidia-container-toolkit
- Host nvidia driver ≥ 525(對應 CUDA 12.1+)

Windows host 額外:
- Docker Desktop 4.25+
- WSL2 + nvidia driver in WSL
- `wsl --update`(需要 Windows 11 22H2+)

### E. Volume mounts

```yaml
volumes:
  - type: bind
    source: ./runs
    target: /app/runs
    read_only: false      # 訓練要寫
  - type: bind
    source: ./.env
    target: /app/.env
    read_only: true       # 防止逃逸後改 key
  - type: bind
    source: ./openspec/changes
    target: /app/openspec/changes
    read_only: true       # 容器內可讀 spec(辯論用),不能改
```

**程式碼不 mount**:程式碼透過 `COPY` 進 image,確保 image 是 immutable 的執行單元。修改程式碼 → rebuild image。

### F. Build reproducibility

`Dockerfile` 開頭釘住 `--platform=linux/amd64` 與 `BuildKit` 必啟用。`pyproject.toml` + `uv.lock`(如果 env-setup spec 已完成遷移)是唯一可變相依來源;每次 build 加 `--no-cache` 確認可從零重建。

CI 未來可加:
- `docker build --label org.opencontainers.image.revision=<git-sha>`
- Image 標 git SHA tag
- Push 到 GitHub Container Registry(免費)讓論文 supplementary 提供 image link

### G. 預設 entry point

```dockerfile
ENTRYPOINT ["python", "-m", "hermes_dqn.training.train"]
```

跑時:
```bash
docker compose run --rm train --reward-source llm --episodes 1500 --seed 42 --memory-db /app/runs/memory.sqlite
```

`tools/run-in-docker.sh` 把上面這串包成單一指令,從 host 端透明跑。

## Risks / Trade-offs

- **Setup 成本**:Windows 上 nvidia-container-toolkit 偶有兼容問題(Docker Desktop 與 WSL2 driver 版本要對齊)。**Mitigation**:`docs/sandbox-architecture.md` 內列出已知 issue 與 fallback(降級到 L2-only)
- **訓練速度**:GPU passthrough 損耗約 5-10%。**Mitigation**:接受;若 demo 對速度敏感,fall back 到 L2-only 跑主機
- **Image 大小**:CUDA runtime image ~3 GB,加上 PyTorch ~5 GB。**Mitigation**:多階段 build + slim image(可選);本 MVP 接受 ~5 GB
- **`.env` 在容器內可讀**:即使是 `:ro` mount,容器內任何程式都看得到 GOOGLE_API_KEY。**Mitigation**:Docker secrets 是 production 路徑,本 change MVP 接受 `:ro` 並在 doc 內標明
- **跨 host 重現性**:不同 NVIDIA driver 版本仍可能造成微小數值差異。**Mitigation**:`config.json` 已記錄 `torch.version.cuda`,差異發生時可追溯
- **容器內仍能 import 全部 numpy**:L1 sandbox 在容器內仍生效,但容器邊界不再依賴 import 限制 ── reward_fn 即使能 import 任何東西,出不去 volume mount 外的檔案

## Migration Plan

本 change 預期維持 proposal-only 狀態。若有日決定 apply:

1. 撰寫 `Dockerfile` + 基本 build 測試(目標 image < 5 GB)
2. 撰寫 `docker-compose.yml`
3. 在 Linux host 測試 build + 容器內 `--episodes 10` smoke
4. 在 Windows + WSL2 host 同樣測試
5. 撰寫 `docs/sandbox-architecture.md` 描述 L1/L2/L3 並更新 README
6. Strict validate + archive

Rollback:刪除 `Dockerfile` / `docker-compose.yml` / wrappers 即可,因為本 change 不修改任何 Python 程式碼。

## Open Questions

- **Apply 時機** ── 本 change 預期維持 proposal-only 至少到課程 final 之後。若教師明確要求或團隊規模擴大,屆時觸發
- **是否要進 GitHub Container Registry** ── 對於課程內部 demo 不需要;若論文公開 supplementary 需要,加 CI workflow 即可
- **L4 是否存在?** ── 一個假想的「跨機器 sandbox」── 但對學生專案來說過度,本 change 不規劃
- **Apple Silicon 支援** ── nvidia-container-toolkit 不支援 macOS;Mac 上想要 sandbox 只能走 L2 + 額外 Python tooling。本 change 預設不支援 macOS host
