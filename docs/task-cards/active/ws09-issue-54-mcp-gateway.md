# Agent Task Card — WS9

## Issue
#<WS9-number> — [Redesign][WS9] MCP gateway and capability mesh

## Parent epic
#42

## Objective
Create the MCP gateway and capability registry as the preferred edge-integration surface.

## In scope
- new `apps/mcp-gateway`
- relevant shared contracts
- config + telemetry integration
- docs/tests

## Out of scope
- unrestricted write tools
- bypassing approvals or telemetry
- burying MCP wrappers directly in unrelated apps

## Required validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Branch
`redesign/ws9-mcp-gateway`

## PR title
`redesign(ws9): add MCP gateway and capability mesh`
