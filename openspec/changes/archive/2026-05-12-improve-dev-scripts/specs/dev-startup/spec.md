## ADDED Requirements

### Requirement: Pull latest code on session start
The script SHALL run `git pull` at the start of every dev session to ensure the working directory is up to date.

#### Scenario: Git repo present
- **WHEN** the project is a git repository
- **THEN** `git pull origin <current-branch>` is executed before any other step

#### Scenario: Not a git repo
- **WHEN** the project is not a git repository
- **THEN** a warning is printed and the script continues without pulling

### Requirement: Initialize OpenSpec on session start
The script SHALL run `openspec init` after git pull to refresh slash commands and skills.

#### Scenario: openspec available
- **WHEN** `openspec` is installed and accessible in PATH
- **THEN** `openspec init` runs silently and slash commands in `.claude/` are refreshed

#### Scenario: openspec not found
- **WHEN** `openspec` is not in PATH
- **THEN** a warning is printed and the script continues

### Requirement: Read latest handover document
The script SHALL find and display the highest-numbered `NN-handover.md` file.

#### Scenario: Handover document exists
- **WHEN** one or more `NN-handover.md` files exist
- **THEN** the highest-numbered file is read and its full content is printed

#### Scenario: No handover document found
- **WHEN** no `NN-handover.md` files exist
- **THEN** a message is shown prompting the developer to create one or use `/opsx:propose`

### Requirement: Suggest next actions from handover
The script SHALL extract and display the `## Next Actions` section from the latest handover document.

#### Scenario: Next Actions section present
- **WHEN** the latest handover contains a `## Next Actions` section
- **THEN** those action items are printed as suggestions

#### Scenario: Next Actions section absent
- **WHEN** the latest handover does not contain `## Next Actions`
- **THEN** a generic prompt is shown to review the handover and plan next steps
