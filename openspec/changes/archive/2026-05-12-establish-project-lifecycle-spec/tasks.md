## 1. Author Spec Artifacts

- [x] 1.1 Draft `proposal.md` (Why / What Changes / Capabilities / Impact)
- [x] 1.2 Draft `design.md` (Context / Goals / Decisions across 5 phases / Risks / Migration / Open Questions)
- [x] 1.3 Author `specs/doc-standards/spec.md`
- [x] 1.4 Author `specs/env-setup/spec.md`
- [x] 1.5 Author `specs/experiments-protocol/spec.md`
- [x] 1.6 Author `specs/evaluation-criteria/spec.md`
- [x] 1.7 Author `specs/final-deliverables/spec.md`

## 2. Validate

- [x] 2.1 Run `openspec validate establish-project-lifecycle-spec --strict` and confirm `valid` output
- [x] 2.2 Cross-check every Requirement has ≥ 1 Scenario with WHEN/THEN bullets and 4-hashtag heading

## 3. Adoption (post-merge, executed by future changes — NOT this change)

- [ ] 3.1 Each subsequent change (`gemma-reward-generator`, `hermes-memory-layer`, `ast-buffer-manager`, `closed-loop-fitness`) MUST cite the spec scenarios it satisfies in its own `proposal.md`
- [ ] 3.2 `02-ending.sh` to add a non-blocking `openspec validate --strict` warning (tracked in `improve-dev-scripts` follow-up)
- [ ] 3.3 Archive this change via `openspec archive establish-project-lifecycle-spec` once 2.1 + 2.2 pass and the next handover is written
