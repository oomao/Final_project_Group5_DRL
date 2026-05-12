# doc-standards Specification

## Purpose
TBD - created by archiving change establish-project-lifecycle-spec. Update Purpose after archive.
## Requirements
### Requirement: OpenSpec Four-Artifact Mandate
Every OpenSpec change SHALL contain `proposal.md`, `tasks.md`, and at least one `specs/<capability>/spec.md`. `design.md` SHALL be required when ANY of the following hold: (a) the change introduces a new external dependency, (b) the change defines a new public interface across subsystems, or (c) the change touches ≥ 2 capabilities. Otherwise `design.md` MAY be omitted.

#### Scenario: Doc-only change skips design.md
- **WHEN** a contributor opens a change that only edits `README.md` and `白話架構介紹.md`
- **THEN** `proposal.md` + `tasks.md` + `specs/doc-standards/spec.md` SHALL be sufficient
- **AND** `design.md` MAY be omitted

#### Scenario: New external dependency requires design.md
- **WHEN** the `gemma-reward-generator` change introduces the `google-genai` PyPI dependency
- **THEN** `design.md` MUST exist in that change
- **AND** the dependency SHALL be justified in its Decisions section

#### Scenario: Cross-capability change requires design.md
- **WHEN** a change modifies both `experiments-protocol` and `evaluation-criteria` specs
- **THEN** `design.md` MUST exist and document the trade-off

### Requirement: Handover Cadence and Structure
Every `02-ending.sh` execution SHALL produce a new `NN-handover.md` following the numbering-rule spec. Each handover MUST contain the 4 sections in this order: `## Summary`, `## Current Status`, `## Next Actions`, `## Open Questions`.

#### Scenario: Session without code changes still writes handover
- **WHEN** a session ends with no commits (discussion only)
- **THEN** `02-ending.sh` SHALL still create the next sequential `NN-handover.md`
- **AND** the `Summary` section MUST explicitly state "no code change"

#### Scenario: Missing required section is rejected
- **WHEN** a contributor creates a handover missing the `Next Actions` section
- **THEN** reviewer SHALL request the section be added before merging

### Requirement: Numbering Rule Scope
The `NN-` prefix from the numbering-rule spec SHALL apply only to sequential flat artifacts (handover documents, ADRs, lab notes). OpenSpec change directory names SHALL use kebab-case without `NN-` prefix.

#### Scenario: Change directory uses kebab-case
- **WHEN** a new change is scaffolded with `openspec new change`
- **THEN** the directory name MUST be kebab-case (e.g. `gemma-reward-generator`)
- **AND** MUST NOT carry an `NN-` prefix

#### Scenario: Handover document uses NN- prefix
- **WHEN** `02-ending.sh` writes a handover after the existing `01-handover.md`
- **THEN** the new file MUST be named `02-handover.md`

### Requirement: Commit Message Convention
Commit messages SHALL begin with one of the prefixes `feat:`, `fix:`, `docs:`, `chore:`, `spec:`, `refactor:`, or `test:`. The first line MUST be ≤ 72 characters. PRs touching an OpenSpec change MUST link to the change directory in the PR body.

#### Scenario: Spec-only commit uses spec: prefix
- **WHEN** a commit only modifies files under `openspec/changes/<name>/specs/`
- **THEN** the commit message MUST start with `spec:`

#### Scenario: Over-length first line is rejected
- **WHEN** a commit's first line exceeds 72 characters
- **THEN** the pre-commit hook SHALL reject the commit

### Requirement: PR Template
A `.github/PULL_REQUEST_TEMPLATE.md` SHALL exist with 4 sections in this order: `## Linked Change`, `## Summary`, `## Spec Scenario Evidence`, `## Validation`. The `Validation` section MUST include the textual output of `openspec validate <change> --strict`.

#### Scenario: PR missing Spec Scenario Evidence is rejected
- **WHEN** a PR is opened without filling in the `Spec Scenario Evidence` section
- **THEN** reviewer SHALL mark the PR as blocked until evidence is provided

### Requirement: Code Comment Language and WHY-only Rule
All in-code comments SHALL be written in English. Comments SHALL explain WHY (intent, trade-off, hardware-specific reasoning), NOT WHAT (which the code itself shows). Exceptions: non-obvious algorithms, hardware-specific lines (e.g. `cudnn.deterministic` for RTX 4090), and safety/security caveats.

#### Scenario: Decorative comment is removed
- **WHEN** a comment reads `# loop through episodes` next to a `for episode in range(...):` line
- **THEN** the reviewer SHALL request the comment be removed

#### Scenario: Chinese comment is rejected
- **WHEN** a contributor adds `# 訓練主迴圈` in a `.py` file
- **THEN** the reviewer SHALL request the comment be rewritten in English

#### Scenario: WHY comment is kept
- **WHEN** a comment reads `# cudnn.deterministic = True so Hermes' fitness comparison across reward iterations stays bit-exact on RTX 4090`
- **THEN** the comment is accepted

### Requirement: Strict Validation Gate
Every PR merging into `main` SHALL pass `openspec validate <change> --strict` with exit code 0. The textual output MUST be pasted into the PR's `Validation` section. `02-ending.sh` SHALL run the same command as a non-blocking warning at session end.

#### Scenario: Strict validation failure blocks merge
- **WHEN** `openspec validate <change> --strict` returns a non-zero exit code
- **THEN** the PR SHALL be marked as blocked and MUST NOT merge

#### Scenario: Ending script warns on validation failure
- **WHEN** `02-ending.sh` runs at session end and validation fails
- **THEN** the script SHALL print a warning but MUST NOT block the commit/push

### Requirement: Plain-Language Architecture Doc Sync
`白話架構介紹.md` SHALL be updated in the same change that modifies (a) the 3 subsystems architecture (Hermes memory / Gemma reward / AST-Buffer) or (b) any of the 4 interface seams between them. Pure tooling, doc-only, or CI changes SHALL NOT require updates.

#### Scenario: Memory layer change updates plain-language doc
- **WHEN** `hermes-memory-layer` modifies the 4-tier memory structure
- **THEN** the same change MUST update the "4 層筆記本" table in `白話架構介紹.md`

#### Scenario: Tooling change does not trigger update
- **WHEN** a change only updates `.pre-commit-config.yaml`
- **THEN** `白話架構介紹.md` SHALL NOT be required to change

### Requirement: Document Language Policy
Narrative documents (`README.md`, `白話架構介紹.md`, `proposal.md`, handover narrative) SHALL be written in Traditional Chinese (繁體中文). Spec scenarios SHALL use English keywords (`WHEN`, `THEN`, `SHALL`, `MUST`, `SHOULD`, `MAY`) with Traditional Chinese description permitted. Code identifiers, filenames, and commit prefixes SHALL be English only.

#### Scenario: Spec keywords stay English in a Chinese-described scenario
- **WHEN** a scenario reads `- **WHEN** 開發者執行 uv sync`
- **THEN** the `WHEN` keyword stays English and the description in Chinese is accepted

#### Scenario: English filename is required
- **WHEN** a contributor adds `源碼/訓練主迴圈.py` to the repo
- **THEN** the reviewer SHALL request a rename to an English path

