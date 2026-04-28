# Integrations placeholders

Connector stubs live here until concrete provider packages are implemented.

## GitHub issue intake (WS13)

WS13 adds a bounded issue-intake foundation in [`src/index.ts`](./src/index.ts) for orchestrator workstreams.

### Contracts

- `GitHubIssueRecord`: normalized incoming issue payload shape
- `GitHubIssueReader`: retrieval boundary (`repo + issueNumber -> issue`)
- `IssueIntake`: stable downstream intake model (`metadata` + `requirements`)

### Parsing coverage

The intake parser normalizes these requirement fields where present:

- parent epic
- objective/goal
- why
- deliverables
- expected code changes
- acceptance criteria
- constraints
- suggested branch
- suggested PR title
- dependencies
- blockers

Missing optional fields are represented as `null` or empty arrays.

### Service

`GitHubIssueIntakeService` provides:

- `fetchIssueIntake(repo, issueNumber)`

This service depends only on the injected `GitHubIssueReader` contract and performs no GitHub mutation.

### Tests and fixtures

- tests: [`src/index.test.ts`](./src/index.test.ts)
- fixtures: [`src/fixtures/github-issues.ts`](./src/fixtures/github-issues.ts)

## Task-card generation and refresh (WS14)

WS14 adds deterministic task-card generation helpers in [`src/index.ts`](./src/index.ts), using WS13 `IssueIntake` as the upstream authority.

### Contracts and helpers

- `TaskCardGeneratorInput`: normalized generation input (issue intake + task-card config)
- `TaskCardRenderModel`: resolved render model for markdown output
- `buildTaskCardFileName(workstreamId, issueNumber, taskCardName)`
- `buildTaskCardPath(workstreamId, issueNumber, taskCardName)`
- `createTaskCardRenderModel(input)`
- `renderTaskCardMarkdown(model)`
- `refreshTaskCardMarkdown(existingMarkdown, nextModel, options)`

### Filename/path convention

Generated task cards follow:

- `wsXX-issue-<issue-number>-<short-kebab-name>.md`

and default to:

- `docs/task-cards/active/<generated-file-name>`

### Refresh behavior

By default, `refreshTaskCardMarkdown` preserves the existing `## Notes for Agent` section from the current task card.

To explicitly replace notes with generated notes, set:

- `overwriteNotesForAgent: true`

This keeps refresh behavior deterministic while avoiding accidental destruction of intentional human-authored notes.
