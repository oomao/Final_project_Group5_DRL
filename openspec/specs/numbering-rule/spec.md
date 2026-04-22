# Numbering Rule

## Purpose

Define a strict sequential two-digit numbering convention for all changes, handover documents, and numbered artifacts in this project to ensure traceability and prevent gaps or collisions.

## Requirements

### Requirement: Two-digit NN- format
All numbered items SHALL use the `NN-` format where `NN` is a zero-padded two-digit integer (e.g., `01-`, `02-`, `13-`).

#### Scenario: Valid format accepted
- **WHEN** a new item is created with a name like `01-handover.md` or `02-add-feature`
- **THEN** it is accepted as correctly formatted

#### Scenario: Invalid format rejected
- **WHEN** a new item is created with a name like `1-handover.md`, `2`, or `001-change`
- **THEN** validation SHALL fail with a format error

### Requirement: Sequential numbering without gaps
Numbers SHALL be assigned sequentially (previous + 1) and MUST NEVER be skipped, reused, reset, or randomized.

#### Scenario: Sequential numbers are valid
- **WHEN** existing items are `01-`, `02-`, `03-` and a new item is created as `04-`
- **THEN** it is accepted as valid

#### Scenario: Skipped number is rejected
- **WHEN** existing items are `01-` and a new item is created as `03-` (skipping `02-`)
- **THEN** validation SHALL fail with a sequential error

#### Scenario: Duplicate number is rejected
- **WHEN** an item numbered `01-` already exists and a new item is created as `01-`
- **THEN** validation SHALL fail with a duplicate error

### Requirement: First item starts at 01-
If no prior numbered items exist, the first item SHALL be numbered `01-`.

#### Scenario: First change numbered correctly
- **WHEN** no prior changes or handover documents exist
- **THEN** the first created item MUST be `01-`

### Requirement: Infer highest number from existing records
If the current highest number is unknown, it SHALL be inferred by scanning existing records before assigning a new number.

#### Scenario: Correct inference before creation
- **WHEN** existing items include `01-handover.md` and `02-handover.md`
- **THEN** the next item SHALL be assigned `03-`
