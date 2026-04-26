# WS06 — Skills and procedural execution engine

Issue: `https://github.com/Deveadra/versa/issues/51`

## Canonical package decision

WS06 establishes one canonical skills package: `@versa/skills` in [`packages/skills`](../../packages/skills).

- `@versa/skills` defines a bounded registry + execution path for reusable procedural skills.
- Skills remain explicit definitions with typed metadata, inputs, outputs, validation, and failure handling.

## Shared contract surface

WS06 adds first-class skill contracts in [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `SkillDefinitionSchema`
- `SkillMetadataSchema`
- `SkillExecutionRequestSchema`
- `SkillExecutionResultSchema`
- `SkillValidationRequirementSchema`
- `SkillFailureHandlingSchema`
- `SkillApprovalRequirementSchema`

These contracts make skill behavior explicit and reusable rather than prompt-only.

## Foundational registry and execution path

`@versa/skills` provides a single execution path via `createSkillRegistry()` in [`packages/skills/src/index.ts`](../../packages/skills/src/index.ts):

- `register(definition, handler)`
- `list()`
- `get(skillIdOrName)`
- `execute(request)`

The registry validates requests/definitions through shared contracts and returns structured execution results with explicit status + validation output.

## Foundational WS06 skills

WS06 seeds 4 bounded deterministic foundational skills:

- `repo_inspection`
- `issue_branch_prep`
- `pr_summary_generation`
- `baseline_report_generation`

Implemented in [`packages/skills/src/index.ts`](../../packages/skills/src/index.ts) through `createFoundationalSkills()`, `createFoundationalSkillHandlers()`, and `createFoundationalSkillRegistry()`.

## Minimal app hook

`@versa/ai` integrates a minimal skill invocation surface in [`apps/ai/src/server.ts`](../../apps/ai/src/server.ts):

- `GET /skills` (enumerate available registered skills)
- `POST /skills/execute` (invoke the shared execution path by `skillId` or `skillName`)

This is additive and intentionally minimal for foundational execution semantics only.

## Validation and tests

- Shared contract tests expanded in [`packages/shared/src/index.test.ts`](../../packages/shared/src/index.test.ts)
- Registry/foundational-skill tests added in [`packages/skills/src/index.test.ts`](../../packages/skills/src/index.test.ts)

## WS06 boundaries

- No self-improvement lab implementation.
- No MCP gateway implementation.
- No broad autonomous agent behavior.
- No destructive skill behavior.
- Existing runtime behavior remains additive and bounded.
