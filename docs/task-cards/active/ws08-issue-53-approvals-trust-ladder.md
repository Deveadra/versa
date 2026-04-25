# Agent Task Card — WS8

## Issue
#<WS8-number> — [Redesign][WS8] Approvals, trust ladder, and action policy

## Parent epic
#42

## Objective
Implement trust levels, approval policy, and enforcement hooks for safe bounded action.

## In scope
- new `packages/approvals`
- `packages/shared`
- enforcement hooks in `apps/core` and/or `apps/ai`
- telemetry linkage
- docs/tests

## Out of scope
- full approval UI
- enabling high-autonomy modes by default
- implicit prompt-only policy logic

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws8-approvals-trust-ladder`

## PR title
`redesign(ws8): add approvals, trust ladder, and action policy`
