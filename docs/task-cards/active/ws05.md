# Agent Task Card — WS5

## Issue
#<WS5-number> — [Redesign][WS5] Workspace state subsystem

## Parent epic
#42

## Objective
Implement durable named workspace state for project continuity.

## In scope
- new `packages/workspaces`
- `packages/shared`
- `packages/database`
- minimal `apps/core` APIs
- docs/tests

## Out of scope
- polished workspace UI
- automatic project discovery
- unrelated memory refactors

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws5-workspace-state`

## PR title
`redesign(ws5): add workspace state subsystem`
