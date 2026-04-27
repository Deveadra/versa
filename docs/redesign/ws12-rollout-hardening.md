# WS12 — Testing, migration, rollout, and documentation hardening

Issue: `https://github.com/Deveadra/versa/issues/57`

## Objective

Harden the redesign workstreams (WS01–WS11) with explicit validation paths, migration rehearsal guidance, rollout checklists, rollback/runbook instructions, and known-limitations documentation.

## Current platform structure snapshot

The current redesign implementation is a TypeScript monorepo layered alongside the legacy Python runtime.

- Apps (`apps/*`)
  - `@versa/core` (`apps/core`)
  - `@versa/ai` (`apps/ai`)
  - `@versa/web` (`apps/web`)
  - `@versa/mcp-gateway` (`apps/mcp-gateway`)
- Packages (`packages/*`)
  - Shared contracts in `@versa/shared`
  - Runtime config in `@versa/config`
  - Foundational subsystems across memory, logging, approvals, bridge, database, etc.
- Monorepo orchestration
  - Root scripts in `package.json`
  - Task graph in `turbo.json`
  - Workspace map in `pnpm-workspace.yaml`

Legacy Python runtime remains intentionally present and is not removed by redesign slices.

## Redesign validation matrix

The matrix below maps each workstream surface to concrete validation paths available in the current repository.

| Workstream | Primary surface | Validation path |
| --- | --- | --- |
| WS01 config governance | `packages/config`, `packages/shared` | `pnpm typecheck`, package tests in monorepo run |
| WS02 telemetry foundation | `packages/logging`, `apps/core`, `apps/ai` | `pnpm test`, runtime route checks (`/health`) |
| WS03 identity doctrine | `packages/shared`, `packages/security`, app policy hooks | `pnpm test`, `pnpm typecheck` |
| WS04 memory gateway | `packages/memory`, `packages/database`, app integration | `pnpm test`, DB flow rehearsal (`pnpm db:reset && pnpm db:migrate && pnpm db:seed`) |
| WS05 workspace state | `packages/workspaces`, core app exposure | `pnpm test`, API route checks |
| WS06 skills engine | `packages/skills`, `apps/ai` | `pnpm test`, route smoke checks (`/skills`) |
| WS07 environment twin | `packages/environment`, `packages/database` | `pnpm test`, API route checks (`/environments`) |
| WS08 approvals trust ladder | `packages/approvals`, policy enforcement paths | `pnpm test`, governance checks via events/skills metadata |
| WS09 MCP gateway | `apps/mcp-gateway`, shared capability contracts | `pnpm test`, route smoke checks (`/mcp/health`, `/mcp/capabilities`) |
| WS10 AI convergence bridge | `packages/bridge`, `apps/ai` bridge routes | `pnpm test`, route smoke checks (`/bridge/health`, `/ai/execute`) |
| WS11 operator console | `apps/web/app/console`, `apps/web/lib/api.ts` | `pnpm test`, web model tests |
| WS12 hardening | repo docs + validation guidance | `pnpm install`, `pnpm lint`, `pnpm typecheck`, `pnpm test` |

## Migration rehearsal (local)

Use this sequence to rehearse migration + baseline startup from a clean local state:

1. Install dependencies:
   - `pnpm install`
2. Reset and rebuild local DB state:
   - `pnpm db:reset`
   - `pnpm db:migrate`
   - `pnpm db:seed`
3. Run hardening quality gates:
   - `pnpm lint`
   - `pnpm typecheck`
   - `pnpm test`
4. Optional runtime startup checks:
   - `pnpm --filter @versa/core dev`
   - `pnpm --filter @versa/ai dev`
   - `pnpm --filter @versa/web dev`
5. Verify core operational routes manually:
   - Core: `/health`, `/events`, `/workspaces`, `/memory`, `/environments`
   - AI: `/health`, `/skills`, `/bridge/health`, `/bridge/capabilities`
   - MCP gateway: `/health`, `/mcp/health`, `/mcp/capabilities`

## Rollout checklist

Before merging a redesign slice or deployment-oriented change:

- [ ] `pnpm install` succeeds on a clean checkout
- [ ] `pnpm lint` passes
- [ ] `pnpm typecheck` passes
- [ ] `pnpm test` passes
- [ ] DB migration rehearsal completed (`db:reset`, `db:migrate`, `db:seed`)
- [ ] Route-level smoke checks completed for impacted services
- [ ] Documentation for changed behavior/contracts updated in `docs/redesign` and/or top-level docs
- [ ] No-touch constraints respected (no broad unrelated cleanup, no legacy runtime deletion)

## Rollback and recovery runbook

If post-change validation fails or local state becomes inconsistent:

1. Stop running local services (`core`, `ai`, `web`, `mcp-gateway`).
2. Rebuild local database state:
   - `pnpm db:reset`
   - `pnpm db:migrate`
   - `pnpm db:seed`
3. Re-run quality gates:
   - `pnpm lint`
   - `pnpm typecheck`
   - `pnpm test`
4. If failures persist, compare behavior against `main` and isolate by subsystem/package.
5. For blocking defects outside the active bounded issue, stop scope expansion and open a dedicated follow-up issue.

## Known limitations and blockers

- CI currently has both `ci.yml` (TypeScript/pnpm quality path) and `ci.yaml` (Python quality path). They represent different validation surfaces and can cause operational ambiguity.
- Top-level `README.md` still contains substantial legacy Python-first guidance; redesign hardening guidance is additive rather than a full documentation rewrite.
- WS12 does not implement new runtime features; it formalizes validation and operational safety guidance for already-delivered workstreams.

## Bounded-scope statement

WS12 is a hardening/documentation/validation slice. It does not authorize unfinished feature work from prior workstreams or broad subsystem rewrites.
