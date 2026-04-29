# WS17 Roo Dispatch and Result Ingestion

Issue: #85
Task card: `docs/task-cards/active/ws17-issue-85-roo-dispatch-result-ingestion.md`

## Purpose

WS17 adds the first typed dispatch/run-record and deterministic result-ingestion layer so Ultron can persist what was sent to Roo and what came back.

## Dispatch Run Record Contract

Implementation lives in `packages/integrations/src/index.ts` via `createRooDispatchRunRecord(input)`.

The dispatch record includes:

- run id
- issue URL and issue number
- task-card path
- base branch and target branch
- dispatch metadata (mode, actor, timestamp)
- artifact paths for:
  - handoff markdown
  - Roo output capture
  - normalized result summary

## Result Ingestion Contract

Implementation lives in `packages/integrations/src/index.ts` via `ingestRooExecutionResult(input)`.

The ingestion result includes:

- run id
- normalized status
- PR-ready summary text (when present)
- validation command results
- changed-file list
- blocker list
- preserved raw output

## Run Status Classification

WS17 classification output is one of:

- `succeeded`
- `failed`
- `blocked`
- `partial`
- `needs-review`

Initial deterministic classification is driven by parsed blockers, validation outcomes, and explicit status indicators in Roo output.

## Semi-Automated Boundary

WS17 intentionally does **not** implement live Roo automation. It models dispatch/result data boundaries so execution can remain semi-automated while preserving durable run history.

## Test Coverage

`packages/integrations/src/index.test.ts` covers:

- deterministic run-record creation
- successful result ingestion
- failed result ingestion
- blocked result ingestion
- incomplete/unstructured output (`needs-review`)
