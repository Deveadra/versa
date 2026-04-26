# Task Cards

This directory contains the execution packets used by AI agents and human operators to implement bounded work in `versa`.

Task cards are part of the operating system of the repo. They are not casual notes.

---

## Purpose

Task cards exist so agents do not have to guess.

A task card translates a GitHub issue into a repo-local execution brief with exact expectations for:

- scope
- branch creation
- validation
- deliverables
- no-touch constraints
- PR preparation

Task cards are the deterministic execution packet for bounded work.

---

## Canonical model

Use these terms consistently:

- **GitHub issue** = tracking record and acceptance authority
- **task card** = execution packet for one bounded implementation slice
- **template** = standard format used to create new task cards
- **base branch** = branch the work branch must be created from
- **branch** = implementation slice branch
- **PR** = proof and merge unit

This is the required operating model for `versa`.

---

## Directory structure

- Task card template:
  - `docs/templates/agent-task-card.md`
- Execution handoff template:
  - `docs/templates/agent-execution-handoff.md`
- Active task cards:
  - `docs/task-cards/active/`
- Archived or completed task cards:
  - `docs/task-cards/archive/`

Only active task cards should be used for current execution.

Archived cards are historical records and must not be reused without review.

---

## Required workflow

### 1. Planning
A human or planner agent creates or updates a GitHub issue.

The issue is the source of truth for:
- goal
- why
- deliverables
- acceptance criteria
- constraints
- suggested branch strategy if relevant
- suggested PR title if relevant

### 2. Task card creation
A human or planner agent creates a task card from the template.

The planner is responsible for determining and writing:
- task card name
- task card file name
- task card path
- base branch
- work branch
- PR title

The executor must not invent these values.

### 3. Execution
An execution agent such as Roo, Codex, Claude Code, or another coding agent reads:
1. the GitHub issue
2. the active task card
3. the relevant repo files

The executor then:
- reads `Base Branch` and `Branch` from the task card
- switches to the base branch
- creates or switches to the work branch
- performs only the bounded work in the card
- runs the required validation
- prepares PR-ready output

### 4. Proof
The execution agent opens or prepares a PR linked to the issue.

The PR must include:
- what changed
- validation run
- known follow-ups
- any migration or risk notes

### 5. Completion
After merge:
- update the task card status
- move the task card to `docs/task-cards/archive/` when appropriate
- keep the GitHub issue and PR as the durable tracking history

---

## Authority order

When there is ambiguity, use this precedence order:

1. explicit human instruction
2. GitHub issue
3. active task card
4. local template conventions

The template is never the authority for scope by itself.

---

## Planner vs executor responsibilities

### Planner responsibilities
Humans and planner agents are responsible for:
- creating the GitHub issue
- creating the task card from the template
- determining the canonical task card name
- determining the canonical task card file name
- determining the base branch
- determining the work branch
- determining the PR title
- ensuring filenames follow convention
- ensuring issue numbers and paths are correct
- keeping cards updated when scope changes

### Executor responsibilities
Execution agents are responsible for:
- reading the linked GitHub issue before changing code
- reading the active task card before changing code
- extracting `Base Branch` and `Branch` from the task card
- implementing only the assigned task card
- staying within the issue boundaries
- validating their changes
- producing PR-ready output linked to the issue
- reporting blockers rather than inventing scope

### Executors must not
Execution agents must not:
- invent new scope
- silently expand into adjacent workstreams
- invent a different branch name
- create the work branch from an arbitrary currently checked-out branch
- delete legacy Python runtime code unless explicitly authorized
- merge placeholder logic without documenting why it exists
- ignore the GitHub issue in favor of only the task card

---

## How templates are used

### Task card template
`docs/templates/agent-task-card.md` is used only to create or refresh task cards.

It is not the active work item.

### Execution handoff template
`docs/templates/agent-execution-handoff.md` is used only to launch an execution agent against an existing task card.

It is not the issue and it is not the task card.

Do not point an execution agent only at the task card template and expect useful results.

Execution agents should be given:
- a GitHub issue number or URL
- a concrete active task card path

The executor should read `Base Branch` and `Branch` from the task card itself.

---

## Task card lifecycle statuses

Use only these statuses:

- `Draft`
- `Active`
- `Blocked`
- `Review`
- `Merged`
- `Archived`
- `Superseded`

Do not invent extra lifecycle labels unless the repo standard changes.

---

## Task card naming convention

### Task card name
The task card name is the short human-readable slug for the bounded work.

Example:

`architecture-baseline`

### Single-card workstream file name
Use:

`ws##-issue-<github-issue-number>-<task-card-name>.md`

Examples:

```txt
ws00-issue-43-architecture-baseline.md
ws01-issue-46-config-governance.md
ws02-issue-47-telemetry-foundation.md
ws03-issue-48-identity-doctrine.md
ws04-issue-49-memory-gateway.md
ws05-issue-50-workspace-state.md
ws06-issue-51-skills-engine.md
ws07-issue-52-environment-twin.md
ws08-issue-53-approvals-trust-ladder.md
ws09-issue-54-mcp-gateway.md
ws10-issue-55-ai-convergence-bridge.md
ws11-issue-56-operator-console.md
ws12-issue-57-rollout-hardening.md
```

### Multi-slice workstream file name

When a workstream is split into multiple bounded execution slices, use:

`ws##-issue-<issue-number>-task-YY-<task-card-name>.md`

Examples:
```txt
ws04-issue-49-task-01-memory-contracts.md
ws04-issue-49-task-02-memory-storage.md
ws04-issue-49-task-03-memory-gateway-api.md
ws09-issue-54-task-01-mcp-app-bootstrap.md
ws09-issue-54-task-02-capability-registry.md
ws09-issue-54-task-03-internal-resource-exposure.md
```

Do not use shortened filenames like `ws02.md`.

---

## Branch naming convention

The planner or human creating the task card is responsible for determining the canonical branch name.

Use this format for the redesign epic:

`redesign/ws##-<task-card-name>`

Examples:
```txt
redesign/ws00-architecture-baseline
redesign/ws01-config-governance
redesign/ws02-telemetry-foundation
redesign/ws03-identity-doctrine
```

Use zero-padded workstream numbers consistently:

- `ws00`
- `ws01`
- `ws02`

Do not mix:

- `ws0`
- `ws1`
- `ws2`

Execution agents must read and use the branch from the task card.
They must not invent a different branch name.

---

### Base branch rule

Every task card must include a `Base Branch`.

Default:

`main`

If a task intentionally depends on another unmerged branch, that dependency must be stated explicitly in the task card.

Execution agents must:

1. switch to the `Base Branch`
2. update it if needed
3. create the work `Branch` from that `Base Branch`

Execution agents must not create the work branch from whatever branch happens to be currently checked out unless that branch exactly matches the task card’s `Base Branch`.

---

## Required fields inside each task card

Every task card must include these fields:

- Issue URL
- Issue
- Issue Title
- Parent Epic
- Workstream
- Task Card ID
- Task Card Name
- Task Card File Name
- Task Card Path
- Status
- Priority
- Agent Type
- Base Branch
- Branch
- PR Title
- Depends On
- Blocks
- Objective
- In Scope
- Out of Scope
- Files/Areas to Inspect First
- Required Approach
- Required Validation
- Deliverables
- No-Touch Constraints
- Acceptance Criteria
- Notes for Agent

### Example header

```md
# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/43
- Issue: #43
- Issue Title: [Redesign][WS0] Establish architecture baseline, contracts, and target package map
- Parent Epic: #42
- Workstream: WS00

- Task Card ID: WS00-ISSUE43
- Task Card Name: architecture-baseline
- Task Card File Name: ws00-issue-43-architecture-baseline.md
- Task Card Path: docs/task-cards/active/ws00-issue-43-architecture-baseline.md

- Status: Active
- Priority: High
- Agent Type: Roo

- Base Branch: main
- Branch: redesign/ws00-architecture-baseline
- PR Title: redesign(ws00): establish architecture baseline and platform contracts

- Depends On: #42
- Blocks: #46, #47, #48, #49, #50, #51, #52, #53, #54, #55, #56, #57
```

---

## How execution agents must use task cards

Execution agents must treat task cards as binding execution instructions.

Before editing, the executor must:

1. read the linked GitHub issue
2. read the task card
3. extract Base Branch and Branch
4. inspect the current repo state
5. summarize the minimal implementation plan

During execution, the executor must:

- preserve working behavior unless the issue explicitly changes it
- stay within the stated scope
- run the required validation commands
- avoid unrelated cleanup or speculative rewrites
- respect all no-touch constraints

After execution, the executor must:

- report files changed
- report commands run
- report validation results
- report blockers honestly
- prepare PR-ready output

---

## Repo rule

One issue should map to one bounded execution slice unless the issue explicitly states that it is split into multiple task cards.

If a workstream becomes too large, split it into sub-cards instead of letting one card become vague.

That rule is mandatory.
