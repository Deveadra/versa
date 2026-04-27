# WS08 — Approvals, trust ladder, and action policy

## Objective

Implement explicit trust levels, approval policy rules, and enforcement outcomes so bounded action can be evaluated through typed policy contracts rather than implicit prompt behavior.

## Canonical package decision

WS08 establishes one canonical approvals package: `@versa/approvals` in [`packages/approvals`](../../packages/approvals).

- `@versa/approvals` defines foundational approval-policy evaluation behavior and deterministic outcomes (`allow`, `require_approval`, `deny`).
- Approval enforcement remains additive and bounded to foundational paths in this workstream.

## Trust ladder contract surface

WS08 adds trust and approvals contracts to [`packages/shared/src/index.ts`](../../packages/shared/src/index.ts):

- `TrustLevelEnum`
- `ActionClassificationSchema`
- `ApprovalRequestSchema`
- `ActionPolicyRuleSchema`
- `ApprovalResultSchema`
- `ApprovalDecisionRecordSchema`
- `ApprovalEnforcementOutcomeSchema`

The trust ladder levels are explicit and ordered:

- `observe`
- `propose`
- `draft`
- `safe-act`
- `bounded-autonomous`

Ordering helpers (`trustLevelRank`, `isTrustLevelAtLeast`, `isTrustLevelAtMost`) make trust checks deterministic and reusable.

## Approval policy behavior (foundation)

`@versa/approvals` provides a minimal policy engine in [`packages/approvals/src/index.ts`](../../packages/approvals/src/index.ts):

- `createApprovalPolicyEngine(rules?)`
  - validates requests with shared schemas
  - matches enabled policy rules by action pattern, impact, and trust level
  - emits one explicit policy outcome (`allow`, `require_approval`, `deny`)
- `defaultApprovalPolicyRules`
  - deny critical-impact actions by default
  - require approval for high-impact execution paths
  - allow bounded medium execution only at or above `safe-act`

This foundational model is intentionally strict and conservative.

## Minimal enforcement path

WS08 wires a minimal enforcement hook in [`apps/ai/src/server.ts`](../../apps/ai/src/server.ts) at `POST /skills/execute`:

- Build approval request metadata from incoming skill execution request
- Evaluate request via `@versa/approvals`
- Return:
  - `403` when policy outcome is `deny`
  - `409` when policy outcome is `require_approval`
  - normal skill execution flow when policy outcome is `allow`

Approval evaluation is gated by `FEATURE_APPROVALS_ENABLED` from [`packages/config/src/index.ts`](../../packages/config/src/index.ts).

## Telemetry linkage

Approval-policy evaluations are logged as structured telemetry events from `apps/ai`:

- `approval.policy.evaluated`

This creates traceable policy records linked to request trace context.

## Why this is not prompt-only policy

Approval behavior in WS08 is code-enforced, schema-validated, and runtime-evaluated:

- trust levels are typed enums
- policy rules are typed contracts
- outcomes are explicit values
- enforcement path is in application request handling

Policy logic does not rely on model instruction text alone.

## WS08 boundaries

- No approval UI implementation
- No high-autonomy mode enabled by default
- No MCP gateway or operator-console implementation work
- No broad runtime rewrites outside foundational approvals path
