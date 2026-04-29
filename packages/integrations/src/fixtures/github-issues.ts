import type { GitHubIssueRecord } from '../index';

export const EPIC_ISSUE_FIXTURE: GitHubIssueRecord = {
  number: 77,
  title: '[Orchestrator] Epic: issue-driven execution loop',
  body: `## Goal

Build the orchestrator loop foundation.

## Why

The orchestrator needs deterministic state across issue intake and downstream packets.

## Deliverables

- orchestration roadmap
- phased workstream issues

## Acceptance Criteria

- workstreams are defined and linked to the epic

## Constraints

- no execution behavior in this epic issue
`,
  state: 'OPEN',
  url: 'https://github.com/Deveadra/versa/issues/77',
  labels: [{ name: 'orchestrator' }, { name: 'epic' }],
  assignees: [{ login: 'Deveadra' }],
};

export const WORKSTREAM_ISSUE_FIXTURE: GitHubIssueRecord = {
  number: 78,
  title: '[Orchestrator][WS13] GitHub issue intake',
  body: `Owner

Parent epic: #77

## Goal

Create the GitHub issue intake layer that allows Ultron to read an issue and extract structured work requirements for downstream task-card and handoff generation.

## Why

The issue is the source of authority for scope, acceptance criteria, constraints, and deliverables.

## Deliverables

- GitHub issue intake contract
- issue metadata normalization
- issue body parser/extractor
- structured requirement model
- validation for required issue fields where possible

## Expected code changes

- new or expanded GitHub integration package/module
- shared issue-intake types/contracts
- orchestrator-facing intake service
- unit tests for parsing common issue shapes

## Acceptance criteria

- an issue can be fetched by repo and issue number
- missing optional fields are represented clearly without crashing
- downstream workstreams can consume a stable IssueIntake object

## Constraints

- do not generate task cards in this workstream
- do not dispatch Roo or execute code

Suggested branch

orchestrator/ws13-github-issue-intake

Suggested PR title

orchestrator(ws13): add GitHub issue intake foundation

Depends On: #77
Blocks: #79, #80
`,
  state: 'OPEN',
  url: 'https://github.com/Deveadra/versa/issues/78',
  labels: [{ name: 'orchestrator' }, { name: 'ws13' }],
  assignees: [{ login: 'Deveadra' }],
};

export const MINIMAL_ISSUE_FIXTURE: GitHubIssueRecord = {
  number: 100,
  title: 'Minimal issue example',
  body: null,
  state: 'OPEN',
  url: 'https://github.com/Deveadra/versa/issues/100',
  labels: [],
  assignees: [],
};

export const WS14_TASK_CARD_ISSUE_FIXTURE: GitHubIssueRecord = {
  number: 79,
  title: '[Orchestrator][WS14] Task-card generator and refresh workflow',
  body: `Owner

Parent epic: #77

## Goal

Create the task-card generation and refresh workflow that converts a normalized GitHub issue into a repo-local task card following the established task-card standard.

## Why

Task cards are the execution packets for AI coding agents.

## Deliverables

- task-card data contract
- task-card Markdown renderer
- task-card filename/path generator
- task-card refresh/update behavior
- validation against required task-card fields
- docs describing generation rules and lifecycle expectations

## Expected code changes

- reuse shared contracts from WS13 where available
- new task-card generation module/package
- shared task-card schemas/contracts
- tests for task-card rendering and field validation
- docs updates for generated task-card workflow

## Acceptance criteria

- a normalized issue intake object can produce a valid task-card Markdown document
- generated task cards include canonical headers, branch, PR title, scope, validation, no-touch constraints, and notes for agent
- generated filenames follow wsXX-issue-<issue-number>-<short-kebab-name>.md
- refresh behavior preserves intentional human edits where practical or clearly documents overwrite behavior
- tests cover the reference task-card format

## Constraints

- do not fetch GitHub issues directly unless using WS13 contracts
- do not generate Roo handoffs in this workstream
- do not dispatch execution agents
- keep implementation additive and bounded to task-card generation

## Suggested branch

orchestrator/ws14-task-card-generator

## Suggested PR title

orchestrator(ws14): add task-card generator and refresh workflow
`,
  state: 'OPEN',
  url: 'https://github.com/Deveadra/versa/issues/79',
  labels: [{ name: 'orchestrator' }, { name: 'ws14' }],
  assignees: [{ login: 'Deveadra' }],
};

export const WS15_ROO_HANDOFF_ISSUE_FIXTURE: GitHubIssueRecord = {
  number: 80,
  title: '[Orchestrator][WS15] Roo executor handoff generator',
  body: `Owner

Parent epic: #77

## Goal

Create the Roo executor handoff generator that converts an issue plus task card into a precise Roo-ready implementation prompt.

## Why

Roo needs a bounded execution packet that includes authority order, branch rules, files to inspect first, required validation, no-touch constraints, and expected output format.

## Deliverables

- Roo handoff contract
- Roo handoff Markdown renderer
- authority-order and scope-boundary section generator
- branch/setup instruction generator
- validation and final-report instruction generator
- tests covering handoff generation from reference task cards

## Expected code changes

- new handoff generation module/package
- shared handoff schemas/contracts
- Roo-specific renderer/template
- tests for deterministic handoff output
- docs describing Roo handoff expectations

## Acceptance criteria

- a task card and issue intake object can produce a Roo-ready handoff
- handoff clearly instructs Roo to inspect the repo before editing
- handoff includes base branch, work branch, task-card path, issue URL, validation commands, no-touch constraints, and final response format
- handoff preserves issue/task-card scope without adding speculative work
- tests cover the reference Roo handoff format

## Constraints

- do not dispatch Roo automatically in this workstream
- do not parse Roo results in this workstream
- do not implement broad multi-agent routing yet
- keep implementation additive and Roo-focused

## Suggested branch

orchestrator/ws15-roo-handoff-generator

## Suggested PR title

orchestrator(ws15): add Roo executor handoff generator
`,
  state: 'OPEN',
  url: 'https://github.com/Deveadra/versa/issues/80',
  labels: [{ name: 'orchestrator' }, { name: 'ws15' }],
  assignees: [{ login: 'Deveadra' }],
};

export const WS17_ROO_OUTPUT_SUCCESS_FIXTURE = `files changed

- packages/integrations/src/index.ts
- packages/integrations/src/index.test.ts

validation results

- pnpm lint: passed
- pnpm typecheck: passed
- pnpm test: passed

blockers, if any

- None

PR-ready summary: Implemented WS17 dispatch records and deterministic Roo result ingestion.

Status: succeeded
`;

export const WS17_ROO_OUTPUT_FAILED_FIXTURE = `files changed

- packages/integrations/src/index.ts

validation results

- pnpm lint: passed
- pnpm typecheck: failed

blockers, if any

- None

PR-ready summary: Type mismatch found in result ingestion parser path.

Status: failed
`;

export const WS17_ROO_OUTPUT_BLOCKED_FIXTURE = `files changed

- packages/integrations/src/index.ts

validation results

- pnpm lint: unknown

blockers, if any

- Missing required secret for downstream command.

PR-ready summary: Execution blocked due to environment prerequisites.

Status: blocked
`;

export const WS17_ROO_OUTPUT_INCOMPLETE_FIXTURE = `Unstructured executor output.
No explicit sections were provided.
Need human follow-up before merging.
`;

export const WS18_ROO_OUTPUT_PARTIAL_FIXTURE = `files changed

- packages/integrations/src/index.ts

validation results

- pnpm lint: passed
- pnpm typecheck: passed
- pnpm test: failed

blockers, if any

- None

known follow-ups

- Re-run flaky integration snapshot test in CI.

risk/migration notes

- No runtime migration required; contract-only additive update.

review notes

- Verify PR body sections align with reviewer checklist.

PR-ready summary: Added WS18 result summary and PR packet generation with partial validation outcome.

Status: partial
`;

export const WS21_ULTRON_HAPPY_PATH_ISSUE_FIXTURE: GitHubIssueRecord = {
  number: 99,
  title: '[Orchestrator][WS21] End-to-end Ultron happy path',
  body: `Owner

Parent epic: #98

## Goal

Implement a true end-to-end happy-path test suite that exercises Ultron’s full issue-to-execution automation chain from issue intake through post-run updates and blocker/follow-up draft generation.

## Why

The current tests prove important pieces, but they do not yet prove the full orchestrator chain.

## Deliverables

- end-to-end orchestrator happy-path test harness
- deterministic seeded inputs / fixtures
- assertions covering chain outputs and state transitions
- docs describing what the happy-path suite proves and what it does not prove

## Acceptance criteria

- one happy-path suite exercises the full issue-to-execution chain end to end
- the suite proves continuity across intake, handoff, sandbox prep, result ingestion, summary generation, workspace/memory update, and follow-up draft generation
- generated artifacts and durable state changes are asserted at the correct boundaries
- the suite is deterministic enough for CI use
- docs clearly explain what is real vs simulated in the suite

## Constraints

- do not broaden this into new product features
- do not rewrite orchestrator modules just to make tests easier
- do not rewrite the legacy Python runtime
- keep the work bounded to confidence for the existing orchestrator chain

## Suggested branch

orchestrator/ws21-ultron-happy-path-e2e

## Suggested PR title

orchestrator(ws21): add end-to-end Ultron happy-path test suite
`,
  state: 'OPEN',
  url: 'https://github.com/Deveadra/versa/issues/99',
  labels: [{ name: 'orchestrator' }, { name: 'ws21' }],
  assignees: [{ login: 'Deveadra' }],
};
