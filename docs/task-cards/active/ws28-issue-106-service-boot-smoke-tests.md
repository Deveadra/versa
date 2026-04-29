# Agent Task Card

- Issue URL: https://github.com/Deveadra/versa/issues/106
- Issue: #106
- Issue Title: [Orchestrator][WS28] Service boot and deployment smoke tests
- Parent Epic: #98
- Workstream: WS28

- Task Card ID: WS28-ISSUE-TBD
- Task Card Name: service-boot-smoke-tests
- Task Card File Name: ws28-issue-tbd-service-boot-smoke-tests.md
- Task Card Path: docs/task-cards/active/ws28-issue-tbd-service-boot-smoke-tests.md

- Status: Draft
- Priority: Medium
- Agent Type: Roo

- Base Branch: main
- Branch: orchestrator/ws28-service-boot-smoke-tests
- PR Title: orchestrator(ws28): add service boot and deployment smoke tests

- Depends On: #98, WS21, WS22
- Blocks: confidence that the platform can boot/run as intended outside isolated unit tests

## Objective

Add service-boot or deployment-smoke tests, if appropriate to current repo reality, so Ultron’s platform surfaces can be validated as bootable together instead of only through isolated tests.

## In Scope

- Inspect how `apps/core`, `apps/ai`, `apps/web`, and any other relevant services are expected to boot in current development/runtime workflows
- Add smoke tests or service-boot checks for supported service combinations where realistic
- Verify basic startup health and configuration wiring for participating services
- Document what is covered, what is intentionally excluded, and how operators should run the smoke checks

## Out of Scope

- Full deployment platform engineering beyond smoke confidence
- Large docker/k8s redesigns if not already present in repo reality
- Broad performance/load testing
- New runtime features unrelated to smoke validation
- Legacy Python rewrite

## Files/Areas to Inspect First

- `apps/core/`
- `apps/ai/`
- `apps/web/`
- `package.json`
- `pnpm-workspace.yaml`
- `turbo.json`
- `.github/workflows/`
- `README.md`
- docs describing local run/start commands

## Required Approach

1. Inspect current repo state before editing.
2. Read the GitHub issue first, then this task card.
3. Be honest about current repo reality: add smoke checks only where boot behavior is meaningful.
4. Prefer lightweight boot/startup validation over brittle pseudo-deployment theater.
5. Keep the PR bounded to smoke confidence work.
6. Document clearly what the smoke checks prove and what they do not prove.

## Required Validation

- `pnpm install`
- `pnpm lint`
- `pnpm typecheck`
- `pnpm test`
- targeted service-boot/smoke command(s)

## Deliverables

- Service boot or smoke checks appropriate to current repo reality
- Any minimal test harness/config needed to run them
- Documentation for operators and future agents on what the checks cover

## No-Touch Constraints

- Do not delete the legacy Python runtime
- Do not perform unrelated repo cleanup
- Do not invent fake deployment complexity just to say smoke tests exist
- Do not drift into full platform deployment engineering
- Do not weaken startup requirements merely to make smoke checks pass

## Acceptance Criteria

- Meaningful service-boot or smoke checks exist for the current platform shape.
- Checks validate startup/config wiring for the relevant services.
- Documentation explains what is covered and what remains outside smoke scope.
- Existing runtime behavior remains intact.

## Notes for Agent

This issue is intentionally last in priority because smoke checks should rest on top of real runtime and boundary tests first. Keep it grounded in what the repo can actually boot today. Read the GitHub issue first, then this task card, then inspect the repo. Rename this file to the canonical issue-numbered pattern after the GitHub issue is created.
