## 1. Status and trigger conditions

This change is intentionally **proposal-only**. The author SHALL NOT begin implementation until one of the trigger conditions in `proposal.md` is met:
- Course supervisor explicitly requests user-uploaded reward execution
- The project is published as a Hugging Face Space / public demo
- Multiple team members share one 4090 with open LLM-generated reward experimentation

- [ ] 1.1 Confirm none of the trigger conditions are met (annual review)
- [ ] 1.2 If triggered, proceed to Task Groups 2–6

## 2. Dockerfile + base image

- [ ] 2.1 Author `Dockerfile` at repo root with `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- [ ] 2.2 Install Python 3.11.x (via deadsnakes PPA or pyenv) — pin patch version aligned with `env-setup` Requirement "Python Version Lock"
- [ ] 2.3 Add `WORKDIR /app`, `COPY pyproject.toml requirements.txt /app/`, then `pip install --no-cache-dir -r requirements.txt`
- [ ] 2.4 `COPY hermes_dqn /app/hermes_dqn`, `COPY tools /app/tools`
- [ ] 2.5 Create `hermes` user (UID 1001) and `USER hermes` before ENTRYPOINT
- [ ] 2.6 `ENTRYPOINT ["python", "-m", "hermes_dqn.training.train"]`
- [ ] 2.7 Add `.dockerignore` excluding `runs/`, `__pycache__/`, `.git/`, `.venv/`

## 3. docker-compose.yml

- [ ] 3.1 Service `train` with `build: .`
- [ ] 3.2 GPU passthrough via `deploy.resources.reservations.devices` (nvidia driver)
- [ ] 3.3 Volume mounts: `./runs:/app/runs`, `./.env:/app/.env:ro`, `./openspec/changes:/app/openspec/changes:ro`
- [ ] 3.4 Network: bridge with egress allowlist (iptables in container init OR external script post-up)
- [ ] 3.5 Restart policy `no` (training is one-shot)

## 4. Wrapper scripts

- [ ] 4.1 `tools/run-in-docker.sh` (bash):
  - 4.1.1 Pre-flight check: `command -v docker` succeeds
  - 4.1.2 Forward `"$@"` as final args after `docker compose run --rm train`
- [ ] 4.2 `tools/run-in-docker.ps1` (PowerShell):
  - 4.2.1 Pre-flight check: `Get-Command docker -ErrorAction SilentlyContinue` not null
  - 4.2.2 Same arg forwarding

## 5. Documentation

- [ ] 5.1 Create `docs/sandbox-architecture.md` with:
  - 5.1.1 L1 / L2 / L3 layer responsibilities
  - 5.1.2 Threat-matrix table (from design.md)
  - 5.1.3 Known issues:
    - Windows + WSL2 + nvidia-container-toolkit version mismatch fallback
    - macOS host not supported (no nvidia)
    - `.env` is `:ro` but readable inside container — Docker secrets is the production upgrade
  - 5.1.4 Quickstart: `tools/run-in-docker.sh --episodes 10 --seed 42`
- [ ] 5.2 Update `README.md` 🚀 快速開始 section with a "Container mode (L3 sandbox)" subsection linking to the new doc
- [ ] 5.3 Update `hermes_dqn/README.md` accordingly

## 6. Smoke + verification

- [ ] 6.1 `docker compose build train` succeeds, image ≤ 6 GB
- [ ] 6.2 `docker compose run --rm --gpus all train --episodes 10 --seed 42` produces full run dir
- [ ] 6.3 Performance: 1500-ep `train --reward-source env --seed 42` wall-time ≤ 110% of bare-metal baseline
- [ ] 6.4 Network allowlist: a `curl https://example.com` from inside the container FAILS
- [ ] 6.5 Network allowlist: a `python -c "from google import genai; genai.Client(...).models.list()"` from inside the container SUCCEEDS
- [ ] 6.6 Volume mount: `np.memmap("/host/anything", mode='w+')` inside container fails (no `/host` mount)
- [ ] 6.7 `openspec validate reward-sandbox-isolation --strict` passes
- [ ] 6.8 Ready to `/opsx:archive`
