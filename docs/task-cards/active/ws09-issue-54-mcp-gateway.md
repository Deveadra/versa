# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/54
- Issue: #54
- Issue Title: [Redesign][WS9] MCP gateway and capability mesh
- Parent Epic: #42
- Workstream: WS09

- Task Card ID: WS09-ISSUE54
- Task Card Name: mcp-gateway
- Task Card File Name: ws09-issue-54-mcp-gateway.md
- Task Card Path: docs/task-cards/active/ws09-issue-54-mcp-gateway.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws09-mcp-gateway
- PR Title: redesign(ws09): add MCP gateway and capability mesh

- Depends On: #43, #46, #47, #48, #49, #50, #51, #52, #53
- Blocks: #55, #56, #57

## Objective

Create the MCP gateway and capability registry as the preferred edge-integration surface.

## In Scope

- Create `apps/mcp-gateway` as the canonical MCP gateway app
- Add relevant shared contracts in `packages/shared` for:
  - capability registry entries
  - MCP resource definitions
  - MCP tool definitions
  - MCP prompt/workflow definitions
  - capability metadata
  - gateway health/status structures
  - registration and lookup results
- Integrate config support needed for the MCP gateway
- Integrate telemetry support needed for gateway activity and traceability
- Add docs describing:
  - MCP gateway purpose
  - capability registry purpose
  - what belongs in the gateway vs unrelated apps
  - how internal capabilities are exposed
  - how external MCP-facing work should be routed through the gateway
- Add tests for foundational gateway/registry behavior
- Keep implementation additive and bounded to MCP gateway and capability-mesh foundations only

## Out of Scope

- Unrestricted write tools
- Bypassing approvals or telemetry
- Burying MCP wrappers directly in unrelated apps
- Broad runtime rewrites unrelated to MCP gateway foundations
- Full external integration rollout across every subsystem
- Frontend/operator-console work
- Unrelated cleanup

## Files/Areas to Inspect First

- `apps/mcp-gateway/`
- `packages/shared/`
- `packages/config/`
- `packages/logging/`
- `apps/core/`
- `apps/ai/`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`
- `docs/task-cards/active/ws01-issue-46-config-governance.md`
- `docs/task-cards/active/ws02-issue-47-telemetry-foundation.md`
- `docs/task-cards/active/ws03-issue-48-identity-doctrine.md`
- `docs/task-cards/active/ws04-issue-49-memory-gateway.md`
- `docs/task-cards/active/ws05-issue-50-workspace-state.md`
- `docs/task-cards/active/ws06-issue-51-skills-engine.md`
- `docs/task-cards/active/ws07-issue-52-environment-twin.md`
- `docs/task-cards/active/ws08-issue-53-approvals-trust-ladder.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #54 first, then this task card.
3. Preserve working behavior unless the issue explicitly changes it.
4. Prefer additive app scaffolding, contracts, config linkage, telemetry linkage, docs, and tests over broad refactors.
5. Keep the MCP gateway as the canonical edge-integration surface.
6. Do not scatter MCP wrappers across unrelated apps.
7. Keep the PR bounded to this issue only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- MCP-gateway-related code changes
- shared contracts for gateway and capability-registry foundations
- config and telemetry linkage required for foundational gateway behavior
- tests for foundational gateway/registry behavior
- docs updates
- brief implementation notes in the PR body

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not rewrite adjacent subsystems unless strictly necessary for this issue
- Do not introduce placeholder MCP gateway code without documenting why it exists
- Do not expose unrestricted write tools
- Do not bypass approvals or telemetry
- Do not bury MCP wrappers directly in unrelated apps
- Do not drift into frontend operator console work or broad AI runtime rewrites in this issue

## Acceptance Criteria

- `apps/mcp-gateway` exists as the canonical MCP gateway app
- Shared gateway and capability-registry contracts are stable and importable
- Config and telemetry integration exist where needed for foundational gateway behavior
- The boundary between gateway responsibilities and unrelated apps is documented clearly
- Foundational gateway/registry tests exist
- Existing runtime behavior remains intact or improves without breakage

## Notes for Agent

This workstream depends on the architecture baseline from WS0, canonical config/dependency governance from WS1, telemetry foundation from WS2, identity/doctrine groundwork from WS3, memory hierarchy/gateway groundwork from WS4, workspace-state groundwork from WS5, skills-engine groundwork from WS6, environment-twin groundwork from WS7, and approvals/trust-ladder groundwork from WS8. Read the GitHub issue first, then this task card, then inspect the repo. Extract `Base Branch` and `Branch` from this card. Create or switch to the work branch from the stated base branch only. Stay strictly within WS09 scope.
