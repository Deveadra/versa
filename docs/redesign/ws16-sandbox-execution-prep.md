# WS16 Sandbox Execution Preparation

Issue: #84
Task card: `docs/task-cards/active/ws16-issue-84-sandbox-execution-prep.md`

## Purpose

WS16 adds a preparation-only planning layer that can be executed before Roo (or another executor) performs implementation work.

This layer does **not** execute commands or dispatch agents. It produces a typed readiness result and a bounded execution plan that can be embedded into:

- a Roo handoff
- a run record

## Execution Prep Contract

Implementation lives in `packages/integrations/src/index.ts` via `prepareSandboxExecution(input)`.

The contract produces:

- status: `ready` or `blocked`
- issues: missing-field or policy readiness failures
- plan:
  - issue URL and number
  - task card path
  - repo path
  - base branch and target branch
  - sandbox strategy
  - validation commands
  - safe command allowlist guidance
  - no-touch boundaries
  - environment twin compatibility summary
  - context embedding targets (`roo_handoff`, `run_record`)

## Sandbox Strategy Model

Allowed strategy values:

- `in_place_branch`
- `git_worktree`
- `dry_run_only`

This is a plan-level decision only in WS16.

## Environment Twin Compatibility

The prep result reports compatibility with existing environment twin contracts by using:

- `environmentTwinRequired`
- `environmentTwinSlug`

If twin context is required and no slug is supplied, the prep result is blocked.

## Safe Command Guidance and No-Touch Boundaries

The prep input requires:

- at least one validation command
- at least one allowlisted safe command
- at least one no-touch boundary

If any are missing, readiness is blocked.

## WS16 Boundaries

This workstream intentionally excludes:

- live command execution
- Roo dispatch
- result ingestion
- autonomous behavior beyond preparation

## Test Coverage

`packages/integrations/src/index.test.ts` includes WS16 tests for:

- normal ready-plan generation
- blocked readiness when required fields are missing
