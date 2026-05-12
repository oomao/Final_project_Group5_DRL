## Why

The current `01-startup.sh` is missing the `openspec init` step, so new contributors don't get openspec slash commands set up automatically on session start. Both scripts also need to be verified against the full requirements.

## What Changes

- Add `openspec init` call to `01-startup.sh` (runs after git pull, before reading handover)
- Ensure startup sequence: pull → openspec init → read handover → suggest next actions
- Verify `02-ending.sh` covers: update tasks.md → archive change → write handover doc → git push

## Capabilities

### New Capabilities

- `dev-startup`: Session startup script that pulls code, initializes openspec, reads handover doc, and suggests next actions
- `dev-ending`: Session ending script that updates tasks.md, archives completed changes, writes next handover doc, and pushes to GitHub

### Modified Capabilities

- none

## Impact

- `01-startup.sh` — add openspec init step
- `02-ending.sh` — verify and align with full requirements
- `package.json` — `dev:start` and `dev:ending` scripts remain unchanged
