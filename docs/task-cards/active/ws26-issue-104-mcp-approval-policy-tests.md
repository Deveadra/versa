# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/104
- Issue: #104
- Issue Title: [Orchestrator][WS26] MCP approval and gateway policy tests
- Parent Epic: #98
- Workstream: WS26

- Task Card ID: WS26-ISSUE-104
- Task Card Name: mcp-approval-policy-tests
- Task Card File Name: ws26-issue-104-mcp-approval-policy-tests.md
- Task Card Path: docs/task-cards/active/ws26-issue-104-mcp-approval-policy-tests.md

- Status: Draft
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws26-mcp-approval-policy-tests
- PR Title: orchestrator(ws26): add MCP approval and gateway policy tests

- Depends On: #98, WS16
- Blocks: trustworthy write-capable MCP exposure and safe autonomy growth

## Objective

Add gateway-policy tests that prove MCP-exposed write actions cannot bypass the trust/approval layer and that write-capable capability exposure stays bounded by policy.

## In Scope

- Inspect current MCP gateway tool exposure and approval enforcement points
- Add tests for read-safe vs write-capable action policy handling
- Prove write actions require the declared trust/approval path
- Cover gateway-policy behavior for approved, denied, missing, and insufficient-trust cases where implemented
- Document the tested policy boundary for MCP-exposed actions

## Out of Scope

- Broad new MCP capability implementation unrelated to policy tests
- Large trust/approval architecture redesign unrelated to test confidence
- New external integrations
- Legacy Python rewrite
- Service boot/deployment smoke work

## Files/Areas to Inspect First

- `apps/mcp-gateway/src/`
- `packages/approvals/`
- `packages/shared/`
- `packages/logging/`
- `packages/telemetry/`
- `apps/ai/src/`
- `apps/core/src/`
- `docs/redesign/`
- `docs/adr/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Test the gateway-policy boundary directly.
4. Cover insufficient-trust and missing-approval scenarios explicitly.
5. Keep the PR bounded to MCP write-policy confidence.
6. Document the tested trust assumptions and remaining gaps.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted MCP/approval policy test command(s)

## Deliverables

- Gateway-policy tests for MCP-exposed write actions
- Coverage for approved, denied, and insufficient-trust scenarios
- Documentation describing the MCP approval boundary

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not broaden this into a new MCP feature epic
- Do not weaken approval policy semantics just to make tests pass
- Do not drift into memory-boundary or CI-threshold work beyond what is necessary

## Acceptance Criteria

- Tests prove MCP-exposed write actions cannot bypass approval/trust checks.
- Policy failures are visible and deterministic.
- Read-safe behavior remains intact while write-capable paths stay governed.
- Documentation explains the tested gateway boundary and trust assumptions.
- Existing behavior remains intact.

## Notes for Agent

MCP exposure is powerful only if it remains governed. This issue exists to prove that write-capable tool exposure cannot sneak around the trust ladder. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
