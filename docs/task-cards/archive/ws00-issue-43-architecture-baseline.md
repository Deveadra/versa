# Agent Task Card — WS0

- Issue URL: https://github.com/Deveadra/versa/issues/43
- Task Card ID: WS00-ISSUE43
- Task Card Path: docs/task-cards/active/ws00-issue-43-architecture-baseline.md
- Issue: #43
- Parent Epic: #42
- Workstream: WS00
- Status: Active
- Priority: High
- Agent Type: Roo
- Branch: redesign/ws0-architecture-baseline
- PR Title: redesign(ws0): establish architecture baseline and platform contracts
- Depends On: #42
- Blocks: #46, #47, #48, #49, #50, #51, #52, #53, #54, #55, #56, #57

## Objective
Define the authoritative architecture baseline for the redesign and codify the platform contracts that all later workstreams must follow.

## In Scope
- Add or update ADR/docs under `docs/adr` and/or `docs/redesign`
- Add shared contract definitions in TypeScript under `packages/shared`
- Add architecture README files for core apps/packages if needed
- Introduce any missing top-level documentation references in `README.md`
- Document package ownership, naming standards, dependency directions, and migration rules

## Out of Scope
- Broad refactors
- Deletion of the legacy Python runtime
- Runtime behavior rewrites
- MCP implementation work beyond defining contracts/interfaces if needed
- Unrelated cleanup

## Files/Areas to Inspect First
- `README.md`
- `package.json`
- `pnpm-workspace.yaml`
- `turbo.json`
- `apps/core/`
- `apps/ai/`
- `apps/web/`
- `packages/shared/`
- `packages/config/`
- `packages/database/`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/`

## Required Validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Acceptance Criteria
- The target architecture is documented clearly enough that multiple agents can work in parallel without ambiguity
- Shared contracts compile and are importable
- Package ownership and responsibility are explicit
- Allowed and forbidden dependency directions are documented
- No application behavior is broken

## No-Touch Constraints
- Do not delete or heavily rewrite the legacy Python runtime
- Do not invent scope outside WS0
- Do not perform unrelated cleanup
- Do not merge placeholder contract code without documenting purpose

## Notes for Agent
Read the GitHub issue first, then this task card, then inspect the repo before editing. Prefer additive documentation and contract work over implementation churn. Stay within WS0 only.
