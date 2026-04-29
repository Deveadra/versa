Issue: `https://github.com/Deveadra/versa/issues/80`
Task card: docs/task-cards/active/ws15-issue-80-roo-handoff-generator.md

You are operating in Versa Executor mode for the `versa` repository.

Required workflow:

1. Read the GitHub issue first.
2. Read the task card second.
3. Extract from the task card:
   - Base Branch
   - Branch
4. Inspect the relevant repo files before making any edits.
5. Switch to the Base Branch first.
6. If needed, update the Base Branch from its remote tracking branch.
7. If the target Branch does not exist locally, create it from the Base Branch and switch to it.
8. If the target Branch already exists locally, switch to it.
9. Summarize the minimal implementation plan before editing.
10. Execute only the assigned task card.
11. Stay strictly within scope.
12. Run every validation command listed in the task card before declaring completion.
13. Report:

- files changed
- commands run
- validation results
- blockers, if any
- a PR-ready summary referencing the issue

Authority order:

1. explicit user instruction
2. linked GitHub issue
3. active task card
4. repo-local conventions

Issue context:

- Issue URL: https://github.com/Deveadra/versa/issues/80
- Issue: #80
- Issue Title: [Orchestrator][WS15] Roo executor handoff generator
- Task Card Path: docs/task-cards/active/ws15-issue-80-roo-handoff-generator.md
- Base Branch: main
- Branch: orchestrator/ws15-roo-handoff-generator

Objective:

Implement the Roo executor handoff generator so Ultron can convert a GitHub issue plus active task card into a precise Roo-ready execution prompt.

In Scope:

- Add a Roo handoff contract
- Render a Roo-ready handoff from issue intake data and task-card data
- Add tests for generated handoff content

Out of Scope:

- Live Roo dispatch
- Result ingestion
- Sandbox worktree creation

Files/Areas to Inspect First:

- `docs/templates/agent-task-card.md`
- `docs/task-cards/active/`
- `packages/shared/`
- `apps/core/src/`
- `apps/ai/src/`

Required Validation:

- pnpm install
- pnpm lint
- pnpm typecheck
- pnpm test

No-Touch Constraints:

- Do not delete or rewrite the legacy Python runtime
- Do not implement actual Roo dispatch in this workstream

Expected Deliverables:

- Roo handoff contract
- handoff renderer
- tests for required prompt sections

Blocker Reporting Rules:

- Report blockers explicitly when validation fails
- Do not expand scope silently to fix unrelated failures

Expected Final Response Format:

- files changed
- commands run
- validation results
- blockers, if any
- PR-ready summary referencing issue #80
