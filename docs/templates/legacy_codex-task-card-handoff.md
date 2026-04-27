You are implementing a bounded redesign slice in the `versa` repository.

Authority order:
1. explicit instructions in this prompt
2. GitHub issue #43
3. the active task card for WS0
4. repo-local conventions

Target repo:
Deveadra/versa

GitHub issue:
#43 — [Redesign][WS0] Establish architecture baseline, contracts, and target package map

Issue URL:
https://github.com/Deveadra/versa/issues/43

Parent epic:
#42

Task card path:
docs/task-cards/active/ws00-issue-43-architecture-baseline.md

Branch:
redesign/ws0-architecture-baseline

PR title:
redesign(ws0): establish architecture baseline and platform contracts

Objective:
Define the authoritative architecture baseline for the redesign and codify the platform contracts that all later workstreams must follow.

In scope:
- add or update ADR/docs under docs/adr and/or docs/redesign
- add shared contract definitions in TypeScript under packages/shared
- add architecture README files for core apps/packages if needed
- introduce any missing top-level documentation references in README.md
- document package ownership, naming standards, dependency directions, and migration rules

Out of scope:
- no broad refactors
- no deletion of legacy Python runtime
- no runtime feature rewrites
- no AI behavior changes
- no MCP implementation work beyond defining contracts/interfaces if needed

Required approach:
1. Inspect current repo state before editing.
2. Preserve working behavior unless explicitly required otherwise.
3. Prefer additive documentation and contract work over implementation churn.
4. Keep changes bounded to WS0.
5. Add or update tests only if contract code requires them.
6. Update docs so future agents can work without ambiguity.

Repo areas to inspect first:
- README.md
- package.json
- pnpm-workspace.yaml
- turbo.json
- apps/core/
- apps/ai/
- apps/web/
- packages/shared/
- packages/config/
- packages/database/
- docs/adr/
- docs/redesign/
- docs/task-cards/

Acceptance criteria:
- target architecture is documented clearly enough that multiple agents can work in parallel without ambiguity
- shared contracts compile and are importable
- package ownership/responsibility is explicit
- allowed and forbidden dependency directions are documented
- no application behavior is broken

Required validation:
- pnpm lint
- pnpm typecheck
- pnpm test

No-touch constraints:
- do not delete or heavily rewrite the legacy Python runtime
- do not invent scope outside WS0
- do not perform unrelated cleanup
- do not merge placeholder contract code without documenting purpose

Deliverables:
- bounded code/doc changes for WS0
- brief implementation summary
- validation results
- suggested PR body that closes #43

Begin by inspecting the relevant repo files and summarizing the minimal change plan before editing.
