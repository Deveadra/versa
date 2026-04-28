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
