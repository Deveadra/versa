import { describe, expect, it } from 'vitest';
import { createApprovalPolicyEngine, defaultApprovalPolicyRules } from './index';

describe('approvals package', () => {
  it('requires approval for draft trust medium-impact skill execution', () => {
    const engine = createApprovalPolicyEngine(defaultApprovalPolicyRules);

    const evaluated = engine.decide({
      requestId: 'apr_10000001',
      requestedAt: new Date().toISOString(),
      actor: 'ai-service',
      trustLevel: 'draft',
      action: 'skills.execute',
      classification: {
        id: 'repo-inspection',
        category: 'execute',
        impact: 'medium',
        reversible: true,
        requiresNetwork: false,
      },
      audit: {
        traceId: 'trace-approvals-1',
        source: 'apps.ai',
        timestamp: new Date().toISOString(),
      },
      context: {
        skillId: 'repo-inspection',
      },
    });

    expect(evaluated.result.outcome).toBe('require_approval');
    expect(evaluated.decision).toBeUndefined();
  });

  it('auto-approves safe-act trust medium-impact skill execution', () => {
    const engine = createApprovalPolicyEngine(defaultApprovalPolicyRules);

    const evaluated = engine.decide({
      requestId: 'apr_10000002',
      requestedAt: new Date().toISOString(),
      actor: 'ai-service',
      trustLevel: 'safe-act',
      action: 'skills.execute',
      classification: {
        id: 'repo-inspection',
        category: 'execute',
        impact: 'medium',
        reversible: true,
        requiresNetwork: false,
      },
      audit: {
        traceId: 'trace-approvals-2',
        source: 'apps.ai',
        timestamp: new Date().toISOString(),
      },
      context: {
        skillId: 'repo-inspection',
      },
    });

    expect(evaluated.result.outcome).toBe('allow');
    expect(evaluated.decision?.decision).toBe('auto_approved');
  });

  it('denies critical-impact actions by default', () => {
    const engine = createApprovalPolicyEngine(defaultApprovalPolicyRules);

    const evaluated = engine.decide({
      requestId: 'apr_10000003',
      requestedAt: new Date().toISOString(),
      actor: 'ai-service',
      trustLevel: 'bounded-autonomous',
      action: 'system.destructive',
      classification: {
        id: 'destructive-op',
        category: 'system',
        impact: 'critical',
        reversible: false,
        requiresNetwork: true,
      },
      audit: {
        traceId: 'trace-approvals-3',
        source: 'apps.ai',
        timestamp: new Date().toISOString(),
      },
      context: {},
    });

    expect(evaluated.result.outcome).toBe('deny');
    expect(evaluated.decision?.decision).toBe('denied');
  });
});
