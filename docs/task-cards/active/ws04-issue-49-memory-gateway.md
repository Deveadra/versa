# Agent Task Card — WS4

## Issue
#<WS4-number> — [Redesign][WS4] Canonical memory hierarchy and gateway

## Parent epic
#42

## Objective
Implement the canonical memory hierarchy and gateway so all durable memory access flows through one governed path.

## In scope
- new `packages/memory`
- `packages/shared`
- `packages/database`
- minimal integration points in `apps/core` and/or `apps/ai`
- docs

## Out of scope
- full legacy Python memory migration
- broad MCP exposure
- UI work

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws4-memory-gateway`

## PR title
`redesign(ws4): implement canonical memory hierarchy and gateway`
