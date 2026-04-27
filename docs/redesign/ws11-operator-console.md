# WS11 — Frontend operator console

Issue: `https://github.com/Deveadra/versa/issues/56`

## Objective

Extend `@versa/web` into a foundational operator-facing console that surfaces real platform state and runtime activity for inspection.

## Current operator-console information architecture

WS11 adds a dedicated operator route at [`/console`](../../apps/web/app/console/page.tsx) with six bounded visibility panels:

- health / status
- workspaces
- memory summaries
- traces / logs
- approvals visibility
- environment overview

The navigation entry is added in [`apps/web/app/layout.tsx`](../../apps/web/app/layout.tsx) so the console is reachable alongside existing views.

## Typed API consumption surfaces

The web API client in [`apps/web/lib/api.ts`](../../apps/web/lib/api.ts) is extended with typed operator-console fetchers:

- core health (`GET /health`)
- core-to-ai probe (`GET /ai/health`)
- events (`GET /events`)
- workspaces (`GET /workspaces`)
- memory (`GET /memory`)
- environments (`GET /environments`)
- ai health (`GET /health` on AI service)
- bridge health (`GET /bridge/health`)
- skills summary (`GET /skills`)

Contract typing is imported from `@versa/shared` (for example `DomainEvent`, `WorkspaceSummary`, `MemoryRecord`, `EnvironmentSummary`, and `BridgeHealthStatus`) to keep consumption stable and explicit.

## Approvals visibility approach

Foundational approvals visibility is derived in [`apps/web/app/console/model.ts`](../../apps/web/app/console/model.ts) by combining:

- governed skills (`metadata.approval.required`)
- approval-related event matches in recent domain events

This keeps the UI additive and useful without introducing backend rewrites.

## Validation footprint

Foundational console behavior coverage is added in [`apps/web/app/console/model.test.ts`](../../apps/web/app/console/model.test.ts) for approval visibility derivation.

## WS11 boundaries

- No full polished product UI implementation
- No ad hoc, untyped fetch pattern bypassing shared contracts
- No broad backend rewrites for frontend convenience
- No legacy Python runtime deletion or heavy rewrite
- No unrelated repository cleanup

