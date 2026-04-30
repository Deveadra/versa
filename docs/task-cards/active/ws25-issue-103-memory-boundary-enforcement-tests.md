# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/103
- Issue: #103
- Issue Title: [Orchestrator][WS25] Canonical memory boundary enforcement tests
- Parent Epic: #98
- Workstream: WS25

- Task Card ID: WS25-ISSUE-103
- Task Card Name: memory-boundary-enforcement-tests
- Task Card File Name: ws25-issue-103-memory-boundary-enforcement-tests.md
- Task Card Path: docs/task-cards/active/ws25-issue-103-memory-boundary-enforcement-tests.md

- Status: Draft
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws25-memory-boundary-enforcement-tests
- PR Title: orchestrator(ws25): enforce canonical memory boundary with tests

- Depends On: #98, WS19
- Blocks: trustworthy memory integrity and long-term self continuity

## Objective

Add contract-boundary tests that prove memory writes cannot bypass the canonical memory service and that the single-memory-gateway rule is actually enforced in practice.

## In Scope

- Inspect current memory write paths and public contracts
- Add tests that prove allowed memory writes flow through the canonical memory service
- Add tests or boundary assertions that direct/ad hoc memory writes are not part of the supported public contract
- Cover app-layer and integration-layer usage where relevant
- Document the canonical memory boundary and what callers must do

## Out of Scope

- Large memory subsystem redesign unrelated to boundary enforcement
- New memory features beyond what tests require
- Approval/MCP policy enforcement work
- Vector retrieval redesign
- Legacy Python rewrite
- Broad orchestrator e2e work outside memory-boundary confidence

## Files/Areas to Inspect First

- `packages/memory/`
- `packages/database/`
- `packages/shared/`
- `apps/ai/src/`
- `apps/core/src/`
- `apps/mcp-gateway/src/`
- `packages/integrations/`
- `docs/redesign/`
- `docs/adr/`

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Treat the single memory gateway as a non-negotiable architectural boundary.
4. Add tests at the contract boundary, not only inside isolated implementation details.
5. Keep the PR bounded to boundary enforcement confidence.
6. Document clearly what is and is not an approved write path.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted memory boundary test command(s)

## Deliverables

- Contract-boundary tests for canonical memory writes
- Negative-path coverage for bypass attempts where practical
- Documentation for approved memory write paths and boundary expectations

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not turn this into a broad memory-system redesign
- Do not silently leave ad hoc write paths undocumented
- Do not drift into approval-policy or CI-threshold work beyond what is necessary

## Acceptance Criteria

- Tests prove canonical memory writes flow through the memory service boundary.
- Unsupported bypass paths are either impossible at the public-contract layer or explicitly tested as forbidden.
- Documentation explains the approved write path.
- Existing behavior remains intact.

## Notes for Agent

The uploaded implementation plan treated the single memory gateway as a hard architectural rule. This issue exists to prove that rule rather than merely describe it. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
