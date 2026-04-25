# Agent Task Card — WS7

## Issue
#<WS7-number> — [Redesign][WS7] Environment twin and system map

## Parent epic
#42

## Objective
Implement the environment twin so Ultron can reason over systems, services, access paths, and procedures.

## In scope
- new `packages/environment`
- `packages/shared`
- `packages/database`
- minimal API or read-path integration
- docs/tests

## Out of scope
- manual ingestion of every real environment entry
- UI-heavy work
- collapsing environment into generic notes

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws7-environment-twin`

## PR title
`redesign(ws7): add environment twin and system map`
