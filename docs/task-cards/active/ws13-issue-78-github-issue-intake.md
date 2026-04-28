# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/78
- Issue: #78
- Issue Title: [Orchestrator][WS13] GitHub issue intake
- Parent Epic: #77
- Workstream: WS13

- Task Card ID: WS13-ISSUE78
- Task Card Name: github-issue-intake
- Task Card File Name: ws13-issue-78-github-issue-intake.md
- Task Card Path: docs/task-cards/active/ws13-issue-78-github-issue-intake.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws13-github-issue-intake
- PR Title: orchestrator(ws13): add GitHub issue intake and work requirement extraction

- Depends On: #77

## Objective

Implement the first intake layer for the orchestrator loop so Ultron can read a GitHub issue and extract structured work requirements for downstream task-card and handoff generation.

## In Scope

- Add a GitHub issue intake contract
- Normalize issue metadata into a stable internal structure
- Extract structured work requirements from issue bodies
- Support fields such as:
  - parent epic
  - goal/objective
  - why
  - deliverables
  - expected code changes
  - acceptance criteria
  - constraints
  - suggested branch
  - suggested PR title
  - dependencies/blockers where present
- Add fixtures for representative epic/workstream issue shapes
- Add tests for issue parsing and normalization
- Add documentation describing issue intake assumptions and authority order

## Out of Scope

- GitHub issue creation or mutation
- Task-card generation
- Roo handoff generation
- Sandbox execution
- Result ingestion
- Blocker issue generation
- Legacy Python runtime deletion or rewrite
- Broad orchestration UI work

## Files/Areas to Inspect First

- `packages/shared/`
- `packages/workspaces/`
- `packages/environment/`
- `packages/memory/`
- `apps/core/src/`
- `apps/ai/src/`
- `docs/templates/agent-task-card.md`
- `docs/task-cards/`
- `docs/redesign/`
- `README.md`

## Required Approach

1. Inspect current repo state before editing.
2. Read issue #78 first, then this task card.
3. Preserve the existing GitHub issue/task-card/PR authority model.
4. Prefer typed contracts and deterministic parsing over ad hoc string handling.
5. Keep parsing tolerant of missing optional sections.
6. Add fixtures and tests before relying on the parser elsewhere.
7. Keep this PR bounded to issue intake only.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`

## Deliverables

- GitHub issue intake contract
- issue metadata normalization logic
- structured work requirement extraction
- representative fixtures
- tests for issue parsing/normalization
- docs describing intake assumptions and authority order

## No-Touch Constraints

- Do not delete or rewrite the legacy Python runtime
- Do not implement task-card generation in this issue
- Do not implement Roo handoff generation in this issue
- Do not create or update GitHub issues from code in this issue
- Do not implement sandbox execution or result ingestion
- Do not perform unrelated cleanup

## Acceptance Criteria

- Given a GitHub issue body, the intake layer extracts objective/goal, why, deliverables, acceptance criteria, constraints, branch, PR title, dependencies, and parent epic where present
- Missing optional fields are handled gracefully
- Required downstream fields are represented in a typed structure
- The established task-card authority model remains intact
- Tests cover at least one epic issue and one workstream issue shape

## Notes for Agent

This is the first implementation slice for the Issue -> Task Card -> Roo Handoff -> Result Summary loop. Keep the work narrow. The goal is not to automate everything yet; the goal is to create the stable issue-intake structure that later workstreams can consume.
