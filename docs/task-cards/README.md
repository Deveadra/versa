# Task Cards

This directory contains the execution packets used by AI agents and human operators to implement bounded work in `versa`.

## Purpose

Task cards exist so agents do not have to guess.

A task card translates a GitHub issue into a repo-local execution brief with exact expectations for branch naming, validation, scope boundaries, and deliverables.

Task cards are part of the operating system of the repo, not casual notes.

---

## Canonical model

Use these terms consistently:

- **GitHub issue** = tracking record and acceptance authority
- **task card** = execution packet for one bounded implementation slice
- **template** = standard format used to create new task cards
- **branch** = implementation slice
- **PR** = proof and merge unit

This is the required operating model for `versa`.

---

## Directory structure

- Template:
  - `docs/templates/agent-task-card.md`
- Active task cards:
  - `docs/task-cards/active/`
- Archived or completed task cards:
  - `docs/task-cards/archive/`

Only active task cards should be used for current execution.

Archived cards are historical records and must not be reused without review.

---

## Branch naming convention

The planner or human creating the task card is responsible for determining the canonical branch name.

Use:

`<epic-topic>/ws##-<task-card-name>`

Example:

`redesign/ws00-architecture-baseline`

Execution agents must read and use this value from the task card.
They must not invent a different branch name.

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
- suggested branch
- suggested PR title

### 2. Execution packet creation
A human or planner agent creates a task card from the template.

The task card must:
- link to the GitHub issue
- restate the objective
- define in-scope and out-of-scope work
- define the Task Card Name
- derive the Task Card File Name
- define Base Branch
- define Branch using the repo naming convention
- specify the branch name
- specify the PR title
- define validation commands
- define no-touch constraints
- identify the repo areas to inspect first

### 3. Implementation
An execution agent such as Codex, Claude, Roo, or another coding agent reads:
1. the GitHub issue
2. the task card
3. the relevant repo files

The agent then performs only the bounded work described there.

### 4. Proof
The execution agent opens a PR linked to the issue.

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
3. task card
4. local template conventions

The template is never the authority for scope by itself.

---

## How agents must use task cards

Execution agents must treat task cards as binding execution instructions.

Agents must:
- read `Base Branch` and `Branch` from the task card
- not invent a different branch name
- read the linked GitHub issue before changing code
- inspect current repo state before editing
- preserve working behavior unless the issue explicitly changes it
- stay within the stated scope
- run the required validation commands
- avoid unrelated cleanup or speculative rewrites
- respect all no-touch constraints

Agents must not:
- invent new scope
- silently expand into adjacent workstreams
- delete legacy Python runtime code unless explicitly authorized
- merge placeholder logic without documenting why it exists
- ignore the GitHub issue in favor of only the task card

---

## How the template is used

`docs/templates/agent-task-card.md` is used only to create or refresh task cards.

It is not the active work item.

Do not point an execution agent only at the template and expect useful results.

Execution agents should be given:
- a GitHub issue number
- a concrete task card path
- optionally a target branch name

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

## Naming convention for task card files

### Single-card workstream
Use:

`wsXX-issue-<github-issue-number>-<short-kebab-name>.md`

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

---

### Multi-slice workstream

When a workstream is split into multiple bounded execution slices, use:

`wsXX-issue-<issue-number>-task-YY-<short-kebab-name>.md`

**Examples:**

```txt
ws04-issue-49-task-01-memory-contracts.md
ws04-issue-49-task-02-memory-storage.md
ws04-issue-49-task-03-memory-gateway-api.md
ws09-issue-54-task-01-mcp-app-bootstrap.md
ws09-issue-54-task-02-capability-registry.md
ws09-issue-54-task-03-internal-resource-exposure.md
```

---

## Required fields inside each task card

Every task card must include these fields:

- Issue URL
- Task Card ID
- Task Card Path
- Issue
- Parent Epic
- Workstream
- Status
- Priority
- Agent Type
- Branch
- PR Title
- Depends On
- Blocks
- Objective
- In Scope
- Out of Scope
- Files/Areas to Inspect First
- Required Validation
- Acceptance Criteria
- No-Touch Constraints
- Notes for Agent

**Example header:**

```md
# Agent Task Card

- Issue URL: <full github issue url>
- Task Card ID: WS04-ISSUE49-TASK01
- Task Card Path: docs/task-cards/active/ws04-issue-49-memory-gateway.md
- Issue: #49
- Parent Epic: #42
- Workstream: WS04
- Status: Active
- Priority: High
- Agent Type: Codex
- Branch: redesign/ws4-memory-gateway
- PR Title: redesign(ws4): implement canonical memory hierarchy and gateway
- Depends On: #43, #46
- Blocks: none
```

---

### Expectations for humans and planner agents

Humans and planner agents are responsible for:

- creating the GitHub issue
- creating the task card from the template
- ensuring filenames follow convention
- ensuring issue numbers and paths are correct
- keeping cards updated when scope changes

### Expectations for execution agents

Execution agents are responsible for:

- implementing only the assigned task card
- staying within the issue boundaries
- validating their changes
- producing a PR that links back to the issue
- reporting blockers rather than inventing scope

---

## Repo rule

One issue should map to one bounded execution slice unless the issue explicitly states that it is split into multiple task cards.

If a workstream becomes too large, split it into sub-cards instead of letting one card become vague.

That rule is **mandatory**.
