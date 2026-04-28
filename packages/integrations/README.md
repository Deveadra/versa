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
