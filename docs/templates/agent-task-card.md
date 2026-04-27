# Agent Task Card — WSX

## Issue
#<issue-number> — <issue-title>
#<issue-url>

## Parent epic
#42

## Objective
<copy the Goal section from the issue>

## In scope
- <explicit files/packages/apps allowed>
- <contracts/docs/tests to add>
- <minimal runtime wiring allowed>

## Out of scope
- <clear no-touch areas>
- <legacy Python rewrite exclusions>
- <UI work if not relevant>
- <MCP work if not relevant>

## Repo areas to inspect first
- `README.md`
- `package.json`
- `pnpm-workspace.yaml`
- `turbo.json`
- `apps/core/...`
- `apps/ai/...`
- `apps/web/...`
- `packages/shared/...`
- `packages/config/...`
- `packages/database/...`
- any relevant `docs/adr` or `docs/redesign` files

## Required approach
1. Inspect current repo state before editing.
2. Preserve working behavior unless the issue explicitly changes it.
3. Prefer additive implementation over broad refactor.
4. Keep contracts typed and importable.
5. Add or update tests for new behavior.
6. Add or update docs for the workstream.
7. Keep the PR bounded to this issue only.

## Required validation
- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- any issue-specific validation steps

## Deliverables
- code changes
- tests
- docs
- brief implementation notes in PR body

## No-touch constraints
- do not delete the legacy Python runtime
- do not perform unrelated repo cleanup
- do not rewrite adjacent subsystems unless strictly necessary for this issue
- do not introduce placeholder code without documenting why it exists

## Branch
`redesign/wsX-<slug>`

## PR title
`redesign(wsX): <summary>`

## PR body checklist
- closes #<issue-number>
- summary of what changed
- validation run
- known follow-ups
- risks or migration notes
