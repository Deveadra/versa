# Local development startup

1. `pnpm install`
2. `pnpm db:reset && pnpm db:migrate && pnpm db:seed`
3. `pnpm --filter @versa/core dev`
4. `pnpm --filter @versa/ai dev` (optional)
5. `pnpm --filter @versa/web dev`
