# Agent Task Card — WS1

- Issue URL: https://github.com/Deveadra/versa/issues/46
- Task Card ID: WS01-ISSUE46
- Task Card Path: docs/task-cards/active/ws01-issue-46-config-governance.md
- Issue: #46
- Parent Epic: #42
- Workstream: WS01
- Status: Active
- Priority: High
- Agent Type: Roo
- Base Branch: main
- Branch: redesign/ws01-config-governance
- PR Title: redesign(ws01): establish canonical config and dependency governance
- Depends On: #43
- Blocks: #47, #48, #49, #50, #51, #52, #53, #54, #55, #56, #57

## Objective
Create a canonical configuration layer and dependency-governance baseline so all new subsystems share one trustworthy source of runtime settings, feature flags, and import rules.

## In Scope
- Expand `@versa/config` schema for app ports, database paths, runtime mode, feature flags, MCP settings, telemetry toggles, and bridge settings
- Document config precedence rules and environment naming conventions
- Add explicit dependency-governance guidance for apps vs packages
- Add import-boundary notes for shared/config packages
- Add validation tests for config parsing and defaults

## Out of Scope
- Deep refactor of legacy Python config
- Broad runtime feature implementation beyond config/governance needs
- Unrelated cleanup
- Architectural work already assigned to WS0

## Files/Areas to Inspect First
- `packages/config/`
- `packages/shared/`
- `apps/core/`
- `apps/ai/`
- `README.md`
- `docs/adr/`
- `docs/redesign/`
- `docs/task-cards/active/ws00-issue-43-architecture-baseline.md`

## Required Validation
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Acceptance Criteria
- TypeScript apps/packages can load one shared validated config surface
- Feature toggles needed by later workstreams are represented explicitly
- Dependency boundaries are documented and practical
- Existing app startup behavior remains intact or improves without breakage

## No-Touch Constraints
- Do not deeply rewrite legacy Python config in this issue
- Do not add broad runtime features beyond config and governance
- Do not drift into telemetry, memory, approvals, or MCP implementation work
- Do not perform unrelated cleanup

## Notes for Agent
WS1 depends on the architecture baseline from WS0. Read issue #46 first, then this task card, then inspect the repo. Prefer additive config/schema/documentation changes with strong defaults.
