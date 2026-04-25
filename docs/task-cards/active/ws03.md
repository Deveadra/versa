# Agent Task Card — WS3

## Issue
#<WS3-number> — [Redesign][WS3] Identity and doctrine subsystem

## Parent epic
#42

## Objective
Create the identity and doctrine subsystem that defines who Aerith/Ultron is operationally and what principles govern behavior.

## In scope
- new `packages/identity` or `packages/doctrine`
- `packages/shared`
- related docs
- minimal config additions if needed

## Out of scope
- full prompt overhaul
- legacy Python personality rewrite
- self-improvement changes

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws3-identity-doctrine`

## PR title
`redesign(ws3): add identity and doctrine subsystem`
