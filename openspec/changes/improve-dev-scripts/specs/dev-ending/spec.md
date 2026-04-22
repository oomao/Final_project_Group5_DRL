## ADDED Requirements

### Requirement: Update tasks.md on session end
The script SHALL ensure `tasks.md` exists and prompt the developer to update it before closing.

#### Scenario: tasks.md exists
- **WHEN** `tasks.md` is present in the project root
- **THEN** the script confirms it is ready and proceeds

#### Scenario: tasks.md missing
- **WHEN** `tasks.md` does not exist
- **THEN** a new `tasks.md` is created with a default template

### Requirement: Archive completed openspec change
The script SHALL run `openspec archive` if an active change is complete.

#### Scenario: openspec available
- **WHEN** `openspec` is installed
- **THEN** the script hints the developer to run `openspec archive` if their change is done

#### Scenario: openspec not found
- **WHEN** `openspec` is not in PATH
- **THEN** a warning is printed and the step is skipped

### Requirement: Write next handover document
The script SHALL create a new `NN-handover.md` with auto-incremented number based on existing handover files.

#### Scenario: Prior handovers exist
- **WHEN** handover files like `01-handover.md`, `02-handover.md` exist
- **THEN** the new file is created as `03-handover.md` (next sequential number)

#### Scenario: No prior handovers
- **WHEN** no `NN-handover.md` files exist
- **THEN** the new file is created as `01-handover.md`

#### Scenario: Handover content
- **WHEN** the handover file is created
- **THEN** it includes: Summary of Changes, Current Status, and Next Actions sections

### Requirement: Push changes to GitHub on session end
The script SHALL stage all changes, commit, and push to the current branch.

#### Scenario: Git repo present
- **WHEN** the project is a git repository
- **THEN** `git add .`, `git commit`, and `git push origin <current-branch>` are executed

#### Scenario: Not a git repo
- **WHEN** the project is not a git repository
- **THEN** a warning is printed and the push step is skipped
