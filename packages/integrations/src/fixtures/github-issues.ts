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
