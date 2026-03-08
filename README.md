# versa monorepo

Phase 0 foundation for a local-first orchestrator.

## Workspace layout

- `apps/web` — Next.js shell with Today/Tasks/Settings pages.
- `apps/core` — HTTP orchestrator with task CRUD + event emission.
- `apps/ai` — AI adapter boundary with placeholder capabilities.
- `packages/shared` — canonical domain entities + event contracts.
- `packages/database` — SQLite migrations, repositories, and seed/reset.
- `packages/{config,logging,security,testing,ui,integrations}` — shared building blocks.
- `docs` — architecture docs, ADRs, roadmap, checklists.
- `infra` — deployment/infrastructure placeholders.
- `scripts` — helper scripts.

## Quick start

```bash
pnpm install
pnpm db:reset
pnpm db:migrate
pnpm db:seed
pnpm --filter @versa/core dev
pnpm --filter @versa/web dev
```

## Root scripts

- `pnpm lint`
- `pnpm format`
- `pnpm typecheck`
- `pnpm test`
- `pnpm db:migrate`

## Commit and PR conventions

- Commits: concise, imperative (`feat(core): add task create route`).
- PRs: include summary, checklist coverage, and test evidence.

## Security guardrails

- Never commit secrets.
- `.env.example` contains placeholders only.
- Sensitive data uses classification from `@versa/security` + shared schemas.
