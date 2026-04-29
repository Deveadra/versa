# WS18 Result Summary and PR Review Packet

Issue: #86
Task card: `docs/task-cards/active/ws18-issue-86-result-summary-pr-packet.md`

## Purpose

WS18 adds deterministic, typed summary and PR packet generation on top of WS17 ingestion so Ultron can convert executor output into a review-ready artifact.

## Contracts

Implementation lives in `packages/integrations/src/index.ts`.

- `buildRooResultSummary(input)` creates a normalized summary with:
  - status
  - issue/task-card/branch metadata
  - changed-files summary (`total`, `files`)
  - validation summary (`passed`, totals, command rows)
  - blockers
  - known follow-ups
  - review notes
  - risk/migration notes
  - missing-data flags
  - PR-ready summary string

- `buildRooPrReviewPacket(input)` produces:
  - status passthrough
  - PR title draft when issue number + title are present
  - PR body draft when issue/task-card/branch metadata is present
  - missing-data flags when data is incomplete

## Example PR Review Packet (Draft)

```md
## Summary
Added WS18 result summary and PR packet generation with partial validation outcome.

## Issue / Task Card
- Issue: https://github.com/Deveadra/versa/issues/86
- Task Card: docs/task-cards/active/ws18-issue-86-result-summary-pr-packet.md
- Branch: orchestrator/ws18-result-summary-pr-packet

## Changed Files
- packages/integrations/src/index.ts

## Validation Results
- pnpm lint: passed (passed)
- pnpm typecheck: passed (passed)
- pnpm test: failed (failed)

## Blockers
- none

## Known Follow-ups
- Re-run flaky integration snapshot test in CI.

## Risk / Migration Notes
- No runtime migration required; contract-only additive update.
```

## Notes

- Missing data is never fabricated; it is surfaced in `missingData` arrays.
- WS18 remains bounded to summary/packet generation and does not open PRs.
