# WS13 — GitHub issue intake

Issue: `https://github.com/Deveadra/versa/issues/78`

## Objective

Add a typed and deterministic issue-intake layer so orchestrator workflows can consume normalized GitHub issue metadata and structured requirement fields.

## Authority order

WS13 preserves the established authority model for downstream orchestration consumers:

1. explicit human instruction
2. GitHub issue
3. active task card
4. repo-local conventions

## Intake surface

The issue-intake API is implemented in [`packages/integrations/src/index.ts`](../../packages/integrations/src/index.ts).

Core additions:

- `GitHubIssueRecord` and `GitHubIssueReader` contracts for issue retrieval boundaries
- `IssueIntake` model with separated `metadata` and `requirements`
- deterministic parsing helpers for section extraction and normalized references
- `GitHubIssueIntakeService` for `repo + issueNumber -> IssueIntake`

## Parsing assumptions

Issue parsing intentionally targets the current orchestrator issue style and remains tolerant of missing optional data.

Supported normalized requirement fields include:

- parent epic
- objective/goal
- why
- deliverables
- expected code changes
- acceptance criteria
- constraints
- suggested branch
- suggested PR title
- dependencies and blockers

Missing sections are represented as `null` or empty arrays so downstream workstreams can consume stable shapes without crash behavior.

## Fixtures and tests

Representative fixtures are added for:

- epic issue shape
- workstream issue shape
- minimal issue shape with missing optional fields

Parser and normalization tests are added in [`packages/integrations/src/index.test.ts`](../../packages/integrations/src/index.test.ts).

## WS13 boundaries

- No task-card generation in this workstream
- No Roo handoff generation in this workstream
- No sandbox execution/result ingestion in this workstream
- No GitHub mutation/issue creation from code in this workstream
- No unrelated cleanup
