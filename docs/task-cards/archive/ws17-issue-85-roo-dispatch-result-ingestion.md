# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/78
- Issue: #78
- Issue Title: [Orchestrator][WS17] Roo dispatch and result ingestion
- Parent Epic: #77
- Workstream: WS17

- Task Card ID: WS17-ISSUE-78
- Task Card Name: roo-dispatch-result-ingestion
- Task Card File Name: ws17-issue-78-roo-dispatch-result-ingestion.md
- Task Card Path: docs/task-cards/active/ws17-issue-78-roo-dispatch-result-ingestion.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws17-roo-dispatch-result-ingestion
- PR Title: orchestrator(ws17): add Roo dispatch records and result ingestion

- Depends On: #77, WS15, WS16

## Objective

Implement the first Roo dispatch and result ingestion layer so Ultron can record execution attempts, capture Roo output, and classify the result of a bounded agent run.

## In Scope

- Define a Roo dispatch/run record contract
- Define an execution result ingestion contract
- Add a run status classification model
- Add parser/normalizer behavior for Roo result output
- Define artifact paths for stored handoff/output/result summaries
- Add tests for result ingestion and status classification
- Document the semi-automated dispatch boundary

## Out of Scope

- Full live Roo automation if not yet available
- PR summary generation beyond raw result data
- Blocker issue creation
- Self-improvement / Dream Mode
- Deleting or rewriting the legacy Python runtime
- Unrelated repo cleanup

## Files/Areas to Inspect First

- `packages/shared/`
- `packages/workspaces/`
- `packages/environment/`
- `packages/logging/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/task-cards/active/ws15-issue-78-roo-handoff-generator.md`
- `docs/task-cards/active/ws16-issue-78-sandbox-execution-prep.md`
- `docs/redesign/`
- `docs/adr/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Reuse WS15 handoff contracts and WS16 sandbox preparation contracts where available.
4. Keep the initial dispatch boundary semi-automated if live Roo integration is not available yet.
5. Represent all run records and result classifications with typed contracts.
6. Prefer deterministic parsers and fixtures before adding LLM-based interpretation.
7. Keep the PR bounded to WS17 only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- Roo dispatch/run record contract
- Execution result ingestion contract
- Run status classification model
- Representative Roo output fixtures
- Tests for success, failed, blocked, partial, and needs-review runs
- Documentation describing dispatch/result ingestion boundaries

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not silently expand task scope when a blocker is found
- Do not introduce placeholder code without documenting why it exists
- Do not drift into broad personal-assistant integrations, MCP expansion, or Dream Mode work

## Acceptance Criteria

- Ultron can create a run record for a Roo handoff.
- Roo output can be ingested from a file/string fixture and normalized into structured result data.
- Result status can be classified as `succeeded`, `failed`, `blocked`, `partial`, or `needs-review`.
- Validation results and changed-file summaries can be represented when present.
- Tests cover success, failure, blocked, and incomplete output cases.
- Existing runtime behavior remains intact.

## Notes for Agent

This workstream is the bridge between generated handoffs and useful run records. Do not overbuild live automation yet. The minimum useful result is a durable, typed representation of what was dispatched and what came back.

Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS17 scope.
