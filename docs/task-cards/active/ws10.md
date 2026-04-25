# Agent Task Card — WS10

## Issue
#<WS10-number> — [Redesign][WS10] AI service convergence and legacy Python bridge

## Parent epic
#42

## Objective
Converge the AI-facing service layer and add a controlled bridge to the legacy Python runtime.

## In scope
- `apps/ai`
- `packages/shared`
- `packages/config`
- optional `packages/bridge`
- docs/tests

## Out of scope
- replacing the Python runtime wholesale
- deleting legacy code
- collapsing all future AI behavior into one oversized service

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- any bridge smoke tests created by the implementation

## Branch
`redesign/ws10-ai-convergence-bridge`

## PR title
`redesign(ws10): converge AI service and add legacy Python bridge`
