# Agent Task Card — WS12

## Issue
#<WS12-number> — [Redesign][WS12] Testing, migration, rollout, and documentation

## Parent epic
#42

## Objective
Harden the redesign through validation, migration rehearsal, rollout guidance, and durable documentation.

## In scope
- docs across `README.md`, `docs/redesign`, `docs/adr`
- validation matrix
- rollout/runbook additions
- tests and CI/verification updates where needed

## Out of scope
- unfinished feature implementation from prior workstreams
- broad architectural rework disguised as documentation
- speculative future-only documentation not grounded in repo state

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws12-rollout-hardening`

## PR title
`redesign(ws12): add testing, migration, rollout, and documentation hardening`
