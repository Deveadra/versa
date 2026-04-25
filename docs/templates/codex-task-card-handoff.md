How new task cards get made

Right now, the process is:

You create or update a GitHub issue.
You copy the task card template.
You fill it in for that issue.
You save it under docs/task-cards/active/ using the repo naming convention.
You optionally generate a Codex handoff using the Codex handoff template.
You paste that handoff into Codex.

---

# Codex Task Card Handoff

Use this template when handing a bounded implementation slice to Codex.

This handoff is the launch packet for execution.
It is not the GitHub issue and it is not the task card template.
It is the instruction set that tells Codex how to use the existing issue and task card correctly.

---

## Purpose

This template exists so Codex can execute work in `versa` without guessing.

A complete Codex handoff should point Codex to:
- the authoritative GitHub issue
- the active task card
- the target branch
- the required validation
- the scope boundaries and no-touch rules

Use this template for implementation work, not for planning-only work.

---

## Usage rules

- Always provide a real GitHub issue number.
- Always provide a real task card path.
- Always provide a real branch name.
- Do not send Codex only the GitHub issue.
- Do not send Codex only the task card template.
- Do not omit validation commands.
- Do not omit no-touch constraints.
- Keep the handoff bounded to one implementation slice.

When there is ambiguity, Codex must follow this authority order:
1. explicit human instruction in the handoff
2. the linked GitHub issue
3. the linked task card
4. repo-local conventions

---

## Template

```txt
You are implementing a bounded implementation slice in the `versa` repository.

Authority order:
1. explicit instructions in this prompt
2. GitHub issue #<ISSUE_NUMBER>
3. the active task card for <WORKSTREAM>
4. repo-local conventions

Target repo:
Deveadra/versa

GitHub issue:
#<ISSUE_NUMBER> — <ISSUE_TITLE>

Issue URL:
<ISSUE_URL>

Parent epic:
#<EPIC_NUMBER>

Task card path:
<ACTIVE_TASK_CARD_PATH>

Branch:
<BRANCH_NAME>

PR title:
<PR_TITLE>

Objective:
<SHORT_OBJECTIVE>

In scope:
- <ALLOWED_CHANGE_1>
- <ALLOWED_CHANGE_2>
- <ALLOWED_CHANGE_3>

Out of scope:
- <OUT_OF_SCOPE_1>
- <OUT_OF_SCOPE_2>
- <OUT_OF_SCOPE_3>

Required approach:
1. Inspect current repo state before editing.
2. Preserve working behavior unless explicitly required otherwise.
3. Prefer additive implementation over broad refactors.
4. Keep changes bounded to this issue only.
5. Add or update tests for new behavior where appropriate.
6. Add or update docs for the workstream.
7. Keep the PR bounded to this issue only.

Repo areas to inspect first:
- README.md
- package.json
- pnpm-workspace.yaml
- turbo.json
- <RELEVANT_APP_OR_PACKAGE_PATH_1>
- <RELEVANT_APP_OR_PACKAGE_PATH_2>
- <RELEVANT_DOC_PATH_1>
- <RELEVANT_DOC_PATH_2>

Acceptance criteria:
- <ACCEPTANCE_CRITERION_1>
- <ACCEPTANCE_CRITERION_2>
- <ACCEPTANCE_CRITERION_3>

Required validation:
- <VALIDATION_COMMAND_1>
- <VALIDATION_COMMAND_2>
- <VALIDATION_COMMAND_3>

No-touch constraints:
- <NO_TOUCH_1>
- <NO_TOUCH_2>
- <NO_TOUCH_3>

Deliverables:
- bounded code and/or doc changes for this issue
- brief implementation summary
- validation results
- suggested PR body that closes #<ISSUE_NUMBER>

Begin by inspecting the relevant repo files and summarizing the minimal change plan before editing.
