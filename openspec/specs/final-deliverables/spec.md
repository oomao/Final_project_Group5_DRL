# final-deliverables Specification

## Purpose
TBD - created by archiving change establish-project-lifecycle-spec. Update Purpose after archive.
## Requirements
### Requirement: Paper Format
The final paper SHALL be written in IEEE two-column format with 8 fixed chapters in this order: Abstract, Introduction, Related Work, Method, Experiments, Discussion, Conclusion, References. Total length SHALL be 8 to 12 two-column pages. Both `.docx` and `.pdf` versions MUST be committed to `docx/` directory.

#### Scenario: Paper filename matches version N
- **WHEN** the paper is bumped to version 3
- **THEN** `docx/論文_第三版.docx` MUST exist
- **AND** `docx/論文_第三版.pdf` MUST exist
- **AND** both files MUST be regenerated from the same source on the same commit

#### Scenario: Page count out of range
- **WHEN** the compiled PDF has fewer than 8 or more than 12 two-column pages
- **THEN** the paper MUST be revised before submission

### Requirement: Paper Chapter Ownership
The 4 team members SHALL each own at least one chapter cluster. Default mapping: Member A owns §1 (Introduction) + §6 (Discussion) + integration; Member B owns §2 (Related Work) + §3.2 (Gemma reward generator); Member C owns §3.1 (Hermes memory) + §3.3 (AST-Buffer); Member D owns §4 (Experiments) + §5 (Results) + all figures. Ownership MUST be recorded in `docx/AUTHORS.md`.

#### Scenario: Authors file lists ownership
- **WHEN** the paper is at any version ≥ 1
- **THEN** `docx/AUTHORS.md` MUST list all 4 members and the chapters each owns
- **AND** every chapter MUST appear in at least one member's ownership list

### Requirement: Presentation Format
The final presentation deck SHALL contain 18 to 22 slides organised into 5 sections in this order: Motivation (3 slides), Architecture (4), Method (6), Experiments (5), Demo + Q&A (2). The deck MUST be provided as both `.pptx` and `.pdf` under `PPT/`.

#### Scenario: Slide count in range
- **WHEN** the presentation reaches the version targeted for oral defense
- **THEN** the slide count MUST be between 18 and 22 inclusive

#### Scenario: Section count is 5
- **WHEN** the deck is reviewed
- **THEN** exactly 5 named sections MUST be identifiable from the slide titles or section markers

### Requirement: Presentation Speaker Allocation
At the oral defense, each of the 4 members SHALL present at least one section, approximately 2 minutes per member. Q&A SHALL be primarily handled by Member A (the integrator), with other members fielding domain-specific questions.

#### Scenario: All members speak
- **WHEN** the oral defense begins
- **THEN** every member MUST be assigned at least one section in the slide deck speaker notes
- **AND** the assignment MUST match the chapter ownership from `docx/AUTHORS.md` wherever a section corresponds to a chapter

### Requirement: README Demo-Day Minimum State
On demo day, `README.md` MUST contain ALL of the following: (a) one-line positioning statement, (b) motivation data table (problem statistics or baseline gap), (c) latest architecture diagram, (d) YouTube video link, (e) install instructions, (f) one-line baseline fitness summary (`mean_reward_last100`, `success_rate`, date).

#### Scenario: Missing baseline fitness line blocks demo
- **WHEN** README lacks the one-line baseline fitness summary
- **THEN** the demo-day checklist MUST mark README as incomplete
- **AND** the demo MUST NOT proceed until the line is added

#### Scenario: All 6 items present
- **WHEN** the demo-day checklist runs
- **THEN** all 6 items above MUST be confirmed present in `README.md`

### Requirement: Demo Video
The demo video SHALL be 4 to 6 minutes long, structured in 3 segments: Motivation (60 seconds), Architecture (90 seconds), Live demo (120 seconds, sped-up screen capture is acceptable). The video MUST be uploaded to YouTube as "unlisted" and the URL MUST appear in `README.md`.

#### Scenario: Video duration in range
- **WHEN** the demo video is finalised
- **THEN** total duration MUST be between 4 and 6 minutes inclusive

#### Scenario: Video privacy setting
- **WHEN** the video is published
- **THEN** the YouTube visibility MUST be `Unlisted`, never `Public` or `Private`

### Requirement: Synchronised Version N
On demo day, the paper, presentation deck, demo video, and architecture diagram MUST share the same version number N. Pure typo fixes SHALL NOT increment N. Version bumps SHALL be triggered ONLY by: (a) an OpenSpec change archive, (b) supervisor review feedback, or (c) a significant new experimental result.

#### Scenario: Typo fix does not bump version
- **WHEN** the paper has only typo or grammar corrections since the last version
- **THEN** the version number MUST NOT increment

#### Scenario: Change archive bumps version
- **WHEN** any OpenSpec change is archived AND its content affects paper/slides/video/diagram
- **THEN** all 4 deliverables MUST be re-rendered at version N+1 in a single coordinated commit

#### Scenario: Inconsistent version on demo day
- **WHEN** demo day arrives and the paper is at N=3 but the slides are at N=2
- **THEN** the inconsistency MUST be resolved before the defense, defaulting to the higher N

### Requirement: Per-Member Deliverable
Each of the 4 members SHALL own at least one stand-alone deliverable that demonstrates their contribution. Default mapping: A owns `README.md` + integration; B owns Paper §2-3.2 + a Gemma reward generation demo; C owns Paper §3.1+§3.3 + a Hermes/AST live demo; D owns Paper §4-5 + experiment figures + the demo video edit. Ownership MUST be referenced from `docx/AUTHORS.md` or `README.md`.

#### Scenario: No deliverable assigned to a member
- **WHEN** the demo-day checklist runs
- **THEN** every member MUST have at least one deliverable listed in `docx/AUTHORS.md`
- **AND** an unassigned member MUST trigger reassignment before demo

