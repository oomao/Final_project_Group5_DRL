## ADDED Requirements

### Requirement: Dockerfile produces a working CUDA-enabled image
The repository SHALL contain a `Dockerfile` at root that builds an image based on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` with Python 3.11.x, all dependencies from `pyproject.toml`, and the `hermes_dqn` package copied in. The resulting image SHALL be capable of running `python -m hermes_dqn.training.train --episodes 10 --seed 42` to completion when launched with GPU access on a host that has nvidia-container-toolkit installed.

#### Scenario: Smoke build on Linux host
- **WHEN** a developer on a Linux host with Docker 24+ and nvidia-container-toolkit runs `docker compose build train`
- **THEN** the build MUST complete without errors
- **AND** the resulting image SHALL be smaller than 6 GB
- **AND** `docker compose run --rm --gpus all train --episodes 10` MUST produce `runs/<timestamp>/` with `episodes.jsonl` (10 rows), `config.json`, and `model_final.pt`

#### Scenario: Smoke build on Windows + WSL2 host
- **WHEN** a developer on Windows 11 22H2+ with Docker Desktop 4.25+ and a WSL2 distro containing nvidia driver ≥ 525 runs the same command
- **THEN** the build MUST complete
- **AND** training MUST use the GPU (verified by `torch.cuda.is_available() == True` printed early in the run)

### Requirement: Container runs as non-root user
The image SHALL `USER hermes` (UID 1001) and the working directory SHALL be `/app`. Root-only operations during build (e.g. apt-get install) SHALL be confined to early build stages and `USER hermes` SHALL be set before the final `ENTRYPOINT`.

#### Scenario: Container processes do not run as root
- **WHEN** `docker compose run train -c 'id -u'` is executed
- **THEN** the output MUST be `1001` (not `0`)

#### Scenario: Mounted runs/ is writable by hermes user
- **WHEN** training writes to `/app/runs/<timestamp>/episodes.jsonl`
- **THEN** the file MUST exist on the host at `./runs/<timestamp>/episodes.jsonl` with ownership matching the host user mapping (per nvidia-container-toolkit chown handling)

### Requirement: Network egress restricted to allowlist
At container runtime, network egress SHALL be restricted to:
- `generativelanguage.googleapis.com:443` (Gemma API)
- DNS resolution (port 53)
- Optional package indices ONLY during build (`pypi.org`, `download.pytorch.org`)

Any other outbound connection SHALL be dropped (via iptables in container init or host-level bridge policy).

#### Scenario: Gemma API call succeeds
- **WHEN** training invokes `LLMRewardClient.generate()` inside the container with a valid GOOGLE_API_KEY
- **THEN** the HTTPS call to `generativelanguage.googleapis.com` MUST succeed
- **AND** the LLM response MUST be received within the SDK's normal timeout

#### Scenario: Arbitrary outbound connection is blocked
- **WHEN** a malicious reward function attempts a TCP connection to `example.com:443`
- **THEN** the connection MUST fail with a network error
- **AND** the training SHALL NOT crash (the reward function may raise; the env wrapper passes the exception up but no data is exfiltrated)

### Requirement: GPU passthrough preserves training throughput
Training inside the container SHALL achieve at least 90% of the equivalent host-native throughput on the same hardware. Measured by 1500-episode wall-time delta vs the bare-metal baseline.

#### Scenario: Container 1500-ep wall-time within budget
- **WHEN** `train --reward-source env --episodes 1500 --seed 42` is run inside the container on RTX 4090
- **THEN** the wall-time MUST be no more than 110% of the bare-metal baseline (currently 24m47s, so ≤ 27m17s in container)

### Requirement: Volume mounts isolate state
The container SHALL mount three host paths and NOTHING else:
- `./runs` → `/app/runs` (read-write, training output)
- `./.env` → `/app/.env` (read-only)
- `./openspec/changes` → `/app/openspec/changes` (read-only, for spec-aware tooling)

Source code SHALL be COPY-ed into the image at build time (immutable execution unit).

#### Scenario: Container cannot write to host home directory
- **WHEN** a malicious reward function tries `np.memmap("/host/Users/...", mode="w+")` (assuming somehow it bypassed L1 sandbox)
- **THEN** the call MUST fail with `FileNotFoundError` or `PermissionError` because `/host` is not mounted

#### Scenario: .env is read-only inside container
- **WHEN** anything inside the container attempts to overwrite `/app/.env`
- **THEN** the write MUST fail with `PermissionError`

### Requirement: Architecture documentation
The repository SHALL contain `docs/sandbox-architecture.md` describing the three sandbox layers (L1 AST/builtins, L2 subprocess, L3 container), the threats each layer addresses, and the explicit residual risks at each level.

#### Scenario: Documentation is committed
- **WHEN** the change is archived
- **THEN** `docs/sandbox-architecture.md` MUST exist
- **AND** MUST contain the threat-matrix table from `design.md`
- **AND** MUST link to `hermes-memory-layer/specs/reward-sandbox/spec.md` and `gemma-reward-generator/specs/llm-reward-client/spec.md` for context

### Requirement: Convenience wrapper scripts
Two wrapper scripts SHALL be provided to make container mode invocable from host shells without typing the full docker compose command:
- `tools/run-in-docker.sh` (bash, for Linux/macOS hosts)
- `tools/run-in-docker.ps1` (PowerShell, for Windows hosts)

Both SHALL forward all positional args to `python -m hermes_dqn.training.train` inside the container.

#### Scenario: Wrapper forwards arguments
- **WHEN** a developer runs `tools/run-in-docker.sh --reward-source llm --episodes 100`
- **THEN** the script MUST invoke `docker compose run --rm train --reward-source llm --episodes 100`

#### Scenario: Wrapper aborts cleanly if Docker not present
- **WHEN** the wrapper is invoked on a host without Docker installed
- **THEN** it MUST exit non-zero with a message naming "Docker not installed; see docs/sandbox-architecture.md"
