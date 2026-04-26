# Agent Execution Handoff

Use this template when launching an execution agent against a bounded task card.

This template is executor-agnostic.
It is for Roo, Codex, Claude Code, or any future execution agent.

## Purpose

This handoff tells the execution agent exactly how to use:
- the GitHub issue
- the active task card
- the repo

The handoff does not invent scope.
The issue and task card remain the authority.

## Usage rules

- Always provide a real GitHub issue number or URL.
- Always provide a real task card path.
- Do not provide only the issue.
- Do not provide only the template.
- The executor must read `Base Branch` and `Branch` from the task card.
- The executor must not invent or derive a different branch name.

## Template

```txt
Issue: <GitHub issue number or full GitHub issue URL>
Task card: docs/task-cards/active/<task-card-file-name>.md

You are operating as the execution agent for the `versa` repository.

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

Hard constraints:
- Do not work outside this issue/task card.
- Do not perform unrelated cleanup.
- Do not delete or heavily rewrite legacy Python runtime code unless the issue explicitly authorizes it.
- Do not create the work branch from the currently checked-out branch unless it exactly matches the task card’s Base Branch.
- Do not commit, push, or open a PR unless explicitly asked.

Begin now by reading the issue and task card, extracting Base Branch and Branch from the task card, switching to the correct base, and then giving the minimal plan before editing.
