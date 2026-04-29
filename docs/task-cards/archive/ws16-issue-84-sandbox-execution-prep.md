# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/84
- Issue: #84
- Issue Title: [Orchestrator][WS16] Sandbox execution preparation
- Parent Epic: #77
- Workstream: WS16

- Task Card ID: WS16-ISSUE84
- Task Card Name: sandbox-execution-prep
- Task Card File Name: ws16-issue-84-sandbox-execution-prep.md
- Task Card Path: docs/task-cards/active/ws16-issue-84-sandbox-execution-prep.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws16-sandbox-execution-prep
- PR Title: orchestrator(ws16): add sandbox execution preparation model

- Depends On: #81, #82, #83

## Objective

Implement sandbox execution preparation so Ultron can create or describe a safe bounded execution context before handing work to Roo or another executor.

## In Scope

- Add a sandbox execution preparation contract
- Add a branch/worktree planning model
- Add environment twin compatibility checks where existing contracts allow
- Add safe command allowlist handoff model
- Add sandbox readiness result structure
- Include repo path, base branch, target branch, validation commands, no-touch boundaries, and sandbox strategy in the execution plan
- Add tests for execution-prep planning
- Add docs describing sandbox preparation and limitations

## Out of Scope

- Actual command execution, except for tests/mocks if needed
- Live Roo dispatch
- Result ingestion
- PR creation
- GitHub issue creation
- Full self-improvement/Dream Mode behavior
- Legacy Python runtime deletion or rewrite

## Files/Areas to Inspect First

- `packages/environment/`
- `packages/shared/`
- `packages/workspaces/`
- `apps/core/src/`
- `apps/ai/src/`
- Any handoff files from WS15
- `docs/redesign/`
- `docs/task-cards/active/`
- `README.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #84 first, then this task card.
3. Reuse environment twin contracts where available.
4. Keep execution preparation separate from execution.
5. Make the plan embeddable in Roo handoffs and run records.
6. Prefer safe, typed planning objects over shell behavior.
7. Keep this PR bounded to sandbox preparation.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- sandbox execution preparation contract
- branch/worktree planning model
- safe command allowlist handoff model
- sandbox readiness/result structure
- tests for normal and missing-field scenarios
- docs describing sandbox prep behavior and limitations

## No-Touch Constraints

- Do not delete or rewrite the legacy Python runtime
- Do not execute real repo-modifying commands in this workstream
- Do not implement full Roo dispatch here
- Do not implement result ingestion here
- Do not implement self-improvement/Dream Mode here
- Do not perform unrelated cleanup

## Acceptance Criteria

- Given a task card, the sandbox prep layer can produce a safe execution plan
- The plan identifies base branch, target branch, repo path, sandbox/worktree strategy, and validation commands
- The plan includes command allowlist guidance and no-touch boundaries
- The plan can be embedded into a Roo handoff or run record
- Tests cover normal and missing-field scenarios

## Notes for Agent

This workstream makes the executor loop safer before any agent starts changing files. Treat sandbox preparation as a planning layer, not a command runner.
