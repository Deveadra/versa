# WS20 Blocker and Follow-up Issue Generation

Issue: [#88](https://github.com/Deveadra/versa/issues/88)

## Purpose

WS20 adds typed blocker classification and linked follow-up issue draft generation so blocked/partial runs can be converted into explicit backlog items without silently expanding the current task scope.

## Reused Inputs

- [`RooResultSummary`](../../packages/integrations/src/index.ts) from WS18
- run linkage/status semantics from WS17/WS19

## New Contracts

Implementation lives in [`packages/integrations/src/index.ts`](../../packages/integrations/src/index.ts).

- `RooBlockerType`
  - `dependency`
  - `missing-contract`
  - `environment`
  - `validation`
  - `scope-expansion`
  - `unknown`
- `RooBlockerClassification`
  - normalized blocker type
  - blocker summary and evidence
  - out-of-scope signal
  - suggested scope
  - acceptance criteria
  - constraints
- `RooFollowUpIssueDraft`
  - title/body draft
  - labels
  - explicit linkage to parent epic, originating issue, task card, branch, run id/status, and blocker type

## New Functions

- `classifyRooExecutionBlockers({ summary })`
  - classifies blockers from summary text
  - adds a validation blocker when validation commands fail and no explicit validation blocker is present
- `buildRooFollowUpIssueDrafts({ summary, parentEpic })`
  - produces review-ready issue drafts
  - uses a safe template with linkage, blocker summary, suggested scope, acceptance criteria, and constraints

## Safety Boundary

- Generates draft content only.
- Does not create GitHub issues automatically.
- Preserves explicit no-touch constraints around scope expansion and legacy runtime protection.

## Coverage

[`packages/integrations/src/index.test.ts`](../../packages/integrations/src/index.test.ts) includes WS20 tests for:

- dependency blockers
- missing-contract blockers
- environment blockers
- validation blockers
- scope-expansion blockers
- linked follow-up issue draft generation from blocked/partial run summaries
