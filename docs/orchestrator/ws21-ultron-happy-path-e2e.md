# WS21 Ultron Happy-Path End-to-End Suite

Issue: #99

This document describes what the WS21 happy-path suite proves for the orchestrator chain and what remains intentionally simulated.

## Suite location

- `packages/integrations/src/index.test.ts`
- `describe('ultron orchestrator happy-path end-to-end chain (WS21)')`

## What this suite proves

The test executes one deterministic happy-path chain using existing WS13–WS20 contracts:

1. Issue intake normalization from a seeded GitHub issue fixture.
2. Task-card render model generation and markdown output.
3. Roo handoff render model generation and markdown output.
4. Sandbox execution preparation readiness output.
5. Roo dispatch run-record creation with deterministic run/artifact paths.
6. Roo result ingestion from structured executor output.
7. Result summary generation.
8. PR review packet generation.
9. Post-run workspace + memory update generation.
10. Follow-up draft generation path (asserted empty on happy path).

The assertions verify continuity across issue number, branch/task-card linkage, run identifiers, validation status, and post-run state outputs.

## What is simulated

- GitHub API calls are fixture-backed (no live GitHub network calls).
- Roo execution is represented by deterministic output fixtures.
- Sandbox provisioning is contract-level planning output, not real process/worktree execution.
- Workspace/memory persistence is contract output validation, not live database mutation in this suite.

## Boundaries and intent

- This suite is a bounded confidence test for orchestrator chain continuity.
- It does not add new orchestrator capabilities.
- It does not replace lower-level unit tests for individual modules.
- It does not validate deploy/runtime infrastructure smoke scenarios.
