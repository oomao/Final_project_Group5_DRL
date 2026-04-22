## Context

Two bash scripts manage the dev session lifecycle: `01-startup.sh` (session start) and `02-ending.sh` (session end). The startup script currently skips `openspec init`, meaning slash commands won't be refreshed on each session. Both scripts are wired into `package.json` as `dev:start` / `dev:ending`.

## Goals / Non-Goals

**Goals:**
- Add `openspec init` to startup so slash commands are always up-to-date
- Confirm ending script covers the full wrap-up sequence

**Non-Goals:**
- Changing the numbering of the scripts (01- / 02- stays)
- Adding CI/CD or automated scheduling

## Decisions

### Add `openspec init` after git pull in startup
`openspec init` refreshes `.claude/commands/` and `.claude/skills/`. Running it on every session start ensures any schema or skill updates are applied without manual intervention.

**Alternative considered:** Run only on first session — rejected because schema updates would be missed.

### Startup sequence (fixed order)
1. `git pull` — get latest code first
2. `openspec init` — refresh slash commands with latest schema
3. Read latest `NN-handover.md` — inform developer of current state
4. Extract & display `## Next Actions` — suggest what to do next

### Ending sequence (verify existing)
1. Check / update `tasks.md`
2. `openspec archive` if active change is complete
3. Write next `NN-handover.md` (auto-increment from last)
4. `git add . && git commit && git push`

## Risks / Trade-offs

- `openspec init` is idempotent — safe to run every session, no side effects
- `git pull` may have merge conflicts → script warns and continues (non-blocking)

## Migration Plan

1. Edit `01-startup.sh`: insert `openspec init` between git pull and handover read
2. Verify `02-ending.sh` matches ending sequence above
3. Run `npm run dev:start` to smoke-test

## Open Questions

- None
