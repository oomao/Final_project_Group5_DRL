# Hermes-DQN Project — Claude Guide

## Dev Session Workflow

**Every session follows this pattern:**

```
開始工作  →  npm run dev:start
結束工作  →  npm run dev:ending
```

### dev:start (`01-startup.sh`)

Runs automatically in order:
1. `git pull` — sync latest code from GitHub
2. `openspec init` — refresh slash commands & skills in `.claude/`
3. Read latest `NN-handover.md` — load current project state
4. Display `## Next Actions` — show what to work on next

### dev:ending (`02-ending.sh`)

Runs automatically in order:
1. Check / create `tasks.md`
2. Hint to run `openspec archive` if current change is complete
3. Write next `NN-handover.md` (auto-incremented)
4. `git add . && git commit && git push`

---

## Numbering Rule

All changes, handover documents, and sequentially numbered files **must** follow the `NN-` format.

- Format: two-digit zero-padded prefix — `01-`, `02-`, `13-`
- Numbers are **never** skipped, reused, reset, or randomized
- First item is always `01-` if none exist
- Infer the next number by scanning existing files before creating

❌ Invalid: `1-handover.md`, `03-` after `01-` (skip), duplicate `01-`
✅ Valid: `01-` → `02-` → `03-` → `04-`

Full spec: [`openspec/specs/numbering-rule/spec.md`](openspec/specs/numbering-rule/spec.md)

---

## OpenSpec Workflow

Use openspec slash commands to manage changes:

```
/opsx:propose   — create a new change (proposal + design + specs + tasks)
/opsx:apply     — implement the change following tasks.md
/opsx:archive   — archive completed change, update main specs
/opsx:explore   — explore codebase before making changes
```

Changes live in `openspec/changes/<name>/` with 4 artifacts:
- `proposal.md` — WHY (motivation)
- `design.md` — HOW (technical decisions)
- `specs/` — WHAT (requirements & scenarios)
- `tasks.md` — implementation checklist

---

## Repository

GitHub: https://github.com/oomao/Final_project_Group5_DRL.git
Branch: `main`
Git user: `csm088220@gmail.com`
