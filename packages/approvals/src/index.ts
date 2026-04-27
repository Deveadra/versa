import {
  ActionPolicyRuleSchema,
  ApprovalDecisionRecordSchema,
  ApprovalRequestSchema,
  ApprovalResultSchema,
  isTrustLevelAtLeast,
  isTrustLevelAtMost,
  type ActionPolicyRule,
  type ApprovalDecisionRecord,
  type ApprovalRequest,
  type ApprovalResult,
} from '@versa/shared';

export type PolicyEvaluationInput = ApprovalRequest;

export type PolicyEvaluationResult = ApprovalResult;

export type ApprovalPolicyEngine = {
  evaluate(input: PolicyEvaluationInput): PolicyEvaluationResult;
  decide(input: PolicyEvaluationInput): {
    result: PolicyEvaluationResult;
    decision: ApprovalDecisionRecord | null;
  };
  listRules(): ActionPolicyRule[];
};

export const defaultApprovalPolicyRules: ActionPolicyRule[] = [
  {
    id: 'policy-default-deny-critical',
    name: 'Deny critical impact actions by default',
    description: 'Critical-impact actions are denied unless explicitly overridden in future policy layers.',
    actionPattern: '*',
    minTrustLevel: 'observe',
    appliesToImpact: ['critical'],
    outcome: 'deny',
    requiresApproval: false,
    rationale: 'Critical actions require explicit future policy expansion and are denied in WS08 foundations.',
    enabled: true,
  },
  {
    id: 'policy-default-require-approval-high-impact',
    name: 'Require approval for high-impact execution',
    description: 'High-impact execution/integration/system actions require operator approval.',
    actionPattern: '*',
    minTrustLevel: 'observe',
    appliesToImpact: ['high'],
    outcome: 'require_approval',
    requiresApproval: true,
    rationale: 'High-impact actions must be explicitly reviewed before execution.',
    enabled: true,
  },
  {
    id: 'policy-skill-execution-safe-act',
    name: 'Allow bounded skill execution at safe-act or above for low/medium impact',
    description:
      'Bounded low/medium skill execution can proceed when trust level is at least safe-act and no stricter rule applies.',
    actionPattern: 'skills.execute',
    minTrustLevel: 'safe-act',
    appliesToImpact: ['low', 'medium'],
    outcome: 'allow',
    requiresApproval: false,
    rationale: 'Safe-act trust permits bounded low/medium execution without operator roundtrip.',
    enabled: true,
  },
  {
    id: 'policy-skill-execution-draft-gate',
    name: 'Require approval for medium execute actions below safe-act',
    description:
      'When trust is below safe-act, medium execute actions require operator approval to preserve bounded autonomy.',
    actionPattern: 'skills.execute',
    minTrustLevel: 'observe',
    maxTrustLevel: 'draft',
    appliesToImpact: ['medium'],
    outcome: 'require_approval',
    requiresApproval: true,
    rationale: 'Medium-impact execution below safe-act is not auto-allowed.',
    enabled: true,
  },
];

const actionMatches = (pattern: string, action: string) => pattern === '*' || pattern === action;

const ruleAppliesToImpact = (rule: ActionPolicyRule, impact: ApprovalRequest['classification']['impact']) =>
  rule.appliesToImpact.length === 0 || rule.appliesToImpact.includes(impact);

const ruleAppliesToTrustLevel = (rule: ActionPolicyRule, trustLevel: ApprovalRequest['trustLevel']) => {
  if (!isTrustLevelAtLeast(trustLevel, rule.minTrustLevel)) return false;
  if (rule.maxTrustLevel && !isTrustLevelAtMost(trustLevel, rule.maxTrustLevel)) return false;
  return true;
};

const decideOutcomeToDecision = (
  outcome: PolicyEvaluationResult['outcome'],
): ApprovalDecisionRecord['decision'] => {
  if (outcome === 'allow') return 'auto_approved';
  if (outcome === 'deny') return 'denied';
  return 'requires_operator';
};

export const createApprovalPolicyEngine = (rules: ActionPolicyRule[] = defaultApprovalPolicyRules) => {
  const normalizedRules = rules.map((rule) => ActionPolicyRuleSchema.parse(rule));

  const evaluate = (input: PolicyEvaluationInput): PolicyEvaluationResult => {
    const request = ApprovalRequestSchema.parse(input);
    const matchedRule = normalizedRules.find(
      (rule) =>
        rule.enabled &&
        actionMatches(rule.actionPattern, request.action) &&
        ruleAppliesToImpact(rule, request.classification.impact) &&
        ruleAppliesToTrustLevel(rule, request.trustLevel),
    );

    if (!matchedRule) {
      return ApprovalResultSchema.parse({
        requestId: request.requestId,
        outcome: 'deny',
        reason: 'No approval policy rule matched request.',
        evaluatedAt: new Date().toISOString(),
        requiresApproval: false,
      });
    }

    return ApprovalResultSchema.parse({
      requestId: request.requestId,
      outcome: matchedRule.outcome,
      reason: matchedRule.rationale,
      policyRuleId: matchedRule.id,
      evaluatedAt: new Date().toISOString(),
      requiresApproval: matchedRule.requiresApproval,
    });
  };

  const decide = (input: PolicyEvaluationInput) => {
    const request = ApprovalRequestSchema.parse(input);
    const result = evaluate(request);

    if (result.outcome === 'allow') {
      const autoDecision = ApprovalDecisionRecordSchema.parse({
        decisionId: `apd_${request.requestId}`,
        requestId: request.requestId,
        decision: decideOutcomeToDecision(result.outcome),
        decidedAt: result.evaluatedAt,
        decidedBy: 'approval-policy-engine',
        reason: result.reason,
        policyRuleId: result.policyRuleId,
        audit: request.audit,
      });

      return { result, decision: autoDecision };
    }

    if (result.outcome === 'deny') {
      const denyDecision = ApprovalDecisionRecordSchema.parse({
        decisionId: `apd_${request.requestId}`,
        requestId: request.requestId,
        decision: decideOutcomeToDecision(result.outcome),
        decidedAt: result.evaluatedAt,
        decidedBy: 'approval-policy-engine',
        reason: result.reason,
        policyRuleId: result.policyRuleId,
        audit: request.audit,
      });

      return { result, decision: denyDecision };
    }

    return { result, decision: null };
  };

  return {
    evaluate,
    decide,
    listRules: () => [...normalizedRules],
  } satisfies ApprovalPolicyEngine;
};

export const classifySkillExecution = (input: {
  action: string;
  skillId?: string;
  skillName?: string;
  risky?: boolean;
}): ApprovalRequest['classification'] => {
  const isHighRisk = input.risky === true;
  return {
    id: input.skillId ?? input.skillName ?? input.action,
    category: 'execute',
    impact: isHighRisk ? 'high' : 'medium',
    reversible: !isHighRisk,
    requiresNetwork: false,
    description: `Skill execution classification for ${input.skillName ?? input.skillId ?? input.action}`,
  };
};

export type {
  ActionClassification,
  ActionPolicyRule,
  ApprovalDecision,
  ApprovalDecisionRecord,
  ApprovalRequest,
  ApprovalResult,
  ApprovalOutcome,
  TrustLevel,
} from '@versa/shared';
