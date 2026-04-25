# Agent Task Card — WS2

## Issue
#<WS2-number> — [Redesign][WS2] Telemetry and observability foundation

## Parent epic
#42

## Objective
Establish the telemetry, logging, and execution-trace foundation required to make all later redesign work observable, debuggable, and auditable.

## In scope
- `packages/logging`
- `packages/shared`
- `apps/core/src/server.ts`
- `apps/ai/src/server.ts`
- telemetry docs under `docs/redesign` and/or `docs/adr`

## Out of scope
- full operator UI
- vendor-specific observability stack rollout
- approvals or MCP implementation beyond needed interfaces

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws2-telemetry-foundation`

## PR title
`redesign(ws2): add telemetry and observability foundation`
