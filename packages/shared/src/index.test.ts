import { describe, expect, it } from 'vitest';
import {
  ActionPolicyRuleSchema,
  ApprovalDecisionRecordSchema,
  ApprovalEnforcementOutcomeSchema,
  ApprovalRequestSchema,
  EnvironmentContextBundleSchema,
  EnvironmentTwinCreateRequestSchema,
  DoctrineSchema,
  DomainEventSchema,
  isTrustLevelAtLeast,
  isTrustLevelAtMost,
  MemoryConsolidationRequestSchema,
  MemoryReadRequestSchema,
  MemoryWriteRequestSchema,
  SkillDefinitionSchema,
  SkillExecutionRequestSchema,
  SkillExecutionResultSchema,
  TelemetryEventSchema,
  TrustLevelEnum,
  WorkspaceCheckpointCreateRequestSchema,
  WorkspaceCreateRequestSchema,
  WorkspaceStatePatchSchema,
} from './index';

describe('DomainEventSchema', () => {
  it('validates a task.created event', () => {
    const parsed = DomainEventSchema.parse({
      eventId: 'evt_12345678',
      eventType: 'task.created',
      actor: 'system',
      timestamp: new Date().toISOString(),
      domain: 'core',
      entityRef: { type: 'task', id: 'tsk_12345678' },
      payload: { title: 'hello' },
      sensitivity: 'internal',
      traceId: 'trace-1',
    });

    expect(parsed.eventType).toBe('task.created');
  });
});

describe('TelemetryEventSchema', () => {
  it('validates a telemetry event with trace conventions', () => {
    const parsed = TelemetryEventSchema.parse({
      eventId: 'evt_87654321',
      eventType: 'http.request.completed',
      level: 'info',
      message: 'request completed',
      timestamp: new Date().toISOString(),
      actor: {
        service: 'core',
        source: 'http',
      },
      context: {
        traceId: 'trace-123',
        correlationId: 'corr-123',
        runId: 'run-123',
        requestId: 'req-123',
      },
      attributes: {
        method: 'GET',
        route: '/health',
        statusCode: 200,
      },
    });

    expect(parsed.context.traceId).toBe('trace-123');
    expect(parsed.level).toBe('info');
  });
});

describe('DoctrineSchema', () => {
  it('validates a doctrine document', () => {
    const parsed = DoctrineSchema.parse({
      doctrineId: 'aerith.ultron',
      version: '1.0.0',
      mission: 'Protect operator intent while executing reliable outcomes.',
      operatorPrinciples: ['Be truthful', 'Prefer reversible actions'],
      responseStyle: {
        tone: 'direct',
        verbosity: 'concise',
        markdownRequired: true,
        citationStyle: 'repo-link',
        forbiddenPhrases: ['Great', 'Certainly'],
      },
      decisionPriorities: ['operator_safety', 'mission_alignment', 'truthfulness'],
      escalationRules: [
        {
          id: 'approval-destructive',
          condition: 'destructive operation is requested',
          severity: 'high',
          action: 'request explicit operator approval',
        },
      ],
      autonomyBoundaries: [
        {
          action: 'git push',
          requiresApproval: true,
          rationale: 'remote side effects require operator confirmation',
        },
      ],
      safetyNoGoActions: [
        {
          id: 'no-unauthorized-destructive-changes',
          rule: 'Do not execute destructive repository actions without explicit authorization.',
          rationale: 'Preserve operator control and recoverability.',
        },
      ],
      ownership: {
        team: 'platform',
        maintainers: ['@deveadra'],
      },
      metadata: {
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        changeSummary: 'initial doctrine baseline',
      },
    });

    expect(parsed.doctrineId).toBe('aerith.ultron');
    expect(parsed.decisionPriorities[0]).toBe('operator_safety');
  });
});

describe('Memory contracts', () => {
  it('validates memory write/read/consolidation contracts', () => {
    const write = MemoryWriteRequestSchema.parse({
      tier: 'episodic',
      summary: 'User completed afternoon workout',
      content: { durationMinutes: 45, type: 'cardio' },
      metadata: {
        confidence: 0.82,
        source: 'manual',
        sensitivity: 'private',
        retention: {
          strategy: 'durable',
          ttlDays: 365,
          decayRate: 0.15,
        },
        provenance: {
          actor: 'core-api',
          traceId: 'trace-memory-1',
          subsystem: 'core',
        },
        tags: ['health', 'habit'],
      },
    });

    const read = MemoryReadRequestSchema.parse({
      text: 'workout',
      tiers: ['episodic', 'semantic'],
      minConfidence: 0.5,
      limit: 10,
    });

    const consolidation = MemoryConsolidationRequestSchema.parse({
      sourceMemoryIds: ['mem_12345678', 'mem_87654321'],
      targetTier: 'semantic',
      summary: 'User tends to exercise in afternoons',
      reason: 'pattern observed over multiple sessions',
      content: { cadence: 'weekly' },
      metadata: write.metadata,
    });

    expect(write.tier).toBe('episodic');
    expect(read.limit).toBe(10);
    expect(consolidation.targetTier).toBe('semantic');
  });
});

describe('Workspace contracts', () => {
  it('validates workspace create/state/checkpoint contracts', () => {
    const create = WorkspaceCreateRequestSchema.parse({
      slug: 'deveadra/versa',
      name: 'Versa',
      repository: 'github.com/Deveadra/versa',
      metadata: {
        owner: 'platform',
        tags: ['redesign', 'ws05'],
        source: 'manual',
      },
      state: {
        currentObjective: 'Ship WS05 workspace-state foundations',
        activeBlockers: [{ description: 'None currently', status: 'mitigated' }],
        recentDecisions: [
          {
            summary: 'Use explicit workspace records and checkpoints',
            decidedAt: new Date().toISOString(),
          },
        ],
        importantFiles: [{ path: 'apps/core/src/server.ts', reason: 'workspace API endpoints' }],
        knownCommands: [{ command: 'pnpm test', description: 'full workspace tests' }],
        validatedProcedures: [{ name: 'monorepo validation', command: 'pnpm lint' }],
        nextRecommendedActions: [{ action: 'Add workspace API read path', priority: 'high' }],
      },
    });

    const patch = WorkspaceStatePatchSchema.parse({
      nextRecommendedActions: [
        {
          action: 'Run WS05 validation suite',
          priority: 'high',
        },
      ],
    });

    const checkpoint = WorkspaceCheckpointCreateRequestSchema.parse({
      summary: 'Validated workspace foundations',
      createdBy: 'orion',
    });

    expect(create.slug).toBe('deveadra/versa');
    expect(patch.nextRecommendedActions?.[0]?.priority).toBe('high');
    expect(checkpoint.createdBy).toBe('orion');
  });
});

describe('Environment twin contracts', () => {
  it('validates environment twin create and context bundle contracts', () => {
    const create = EnvironmentTwinCreateRequestSchema.parse({
      slug: 'prod/us-east-1',
      name: 'Production US East',
      metadata: {
        owner: 'platform',
        tags: ['production', 'ws07'],
        source: 'manual',
      },
    });

    const context = EnvironmentContextBundleSchema.parse({
      environment: {
        id: 'env_12345678',
        slug: create.slug,
        name: create.name,
        metadata: {
          owner: 'platform',
          tags: ['production'],
          source: 'manual',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      },
      records: [
        {
          id: 'enr_12345678',
          environmentId: 'env_12345678',
          kind: 'service',
          name: 'core-api',
          attributes: {
            port: 4010,
          },
          metadata: {
            source: 'manual',
            tags: ['core', 'api'],
            confidence: 0.9,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
        },
      ],
      relationships: [
        {
          id: 'enl_12345678',
          environmentId: 'env_12345678',
          fromEntityId: 'enr_12345678',
          toEntityId: 'enr_87654321',
          relation: 'depends_on',
          direction: 'directed',
          createdAt: new Date().toISOString(),
        },
      ],
      accessPaths: [
        {
          id: 'eap_12345678',
          environmentId: 'env_12345678',
          entityId: 'enr_12345678',
          name: 'core api local',
          method: 'http',
          endpoint: 'http://localhost:4010/health',
          prerequisites: ['core server running'],
          commandRefIds: [],
          createdAt: new Date().toISOString(),
        },
      ],
      procedures: [
        {
          id: 'epr_12345678',
          environmentId: 'env_12345678',
          name: 'restart core api',
          intent: 'recover core service after config update',
          targetEntityIds: ['enr_12345678'],
          steps: [
            {
              order: 1,
              instruction: 'run pnpm --filter @versa/core dev',
              expectedOutcome: 'core service starts and health endpoint returns ok',
            },
          ],
          tags: ['recovery'],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        },
      ],
    });

    expect(create.slug).toBe('prod/us-east-1');
    expect(context.records[0]?.kind).toBe('service');
    expect(context.procedures[0]?.steps[0]?.order).toBe(1);
  });
});

describe('Skill contracts', () => {
  it('validates skill definition, execution request, and execution result contracts', () => {
    const definition = SkillDefinitionSchema.parse({
      id: 'repo-inspection',
      name: 'repo_inspection',
      bounded: true,
      deterministic: true,
      tags: ['inspection'],
      metadata: {
        description: 'Inspect repo paths in a bounded way',
        version: '0.1.0',
        inputs: [
          {
            name: 'files',
            description: 'list of files',
            required: true,
          },
        ],
        outputs: [
          {
            name: 'summary',
            description: 'inspection summary',
          },
        ],
        requiredTools: ['read_file'],
        requiredResources: ['workspace'],
        validationChecks: [
          {
            id: 'inputs.files.present',
            description: 'files input provided',
            required: true,
          },
        ],
        failureHandling: {
          retryable: false,
          maxRetries: 0,
        },
        approval: {
          required: false,
        },
      },
    });

    const request = SkillExecutionRequestSchema.parse({
      skillId: definition.id,
      input: {
        files: ['packages/shared/src/index.ts'],
      },
      context: {
        traceId: 'trace-skill-1',
        actor: 'orion',
      },
    });

    const result = SkillExecutionResultSchema.parse({
      executionId: 'skx_12345678',
      skillId: definition.id,
      skillName: definition.name,
      status: 'succeeded',
      startedAt: new Date().toISOString(),
      completedAt: new Date().toISOString(),
      output: {
        summary: 'ok',
      },
      validation: {
        passed: true,
        checks: [
          {
            id: 'inputs.files.present',
            passed: true,
            message: 'files input provided',
          },
        ],
      },
    });

    expect(request.skillId).toBe('repo-inspection');
    expect(result.status).toBe('succeeded');
  });
});

describe('Approval and trust-ladder contracts', () => {
  it('validates trust level enum and ordering helpers', () => {
    const trust = TrustLevelEnum.parse('safe-act');

    expect(trust).toBe('safe-act');
    expect(isTrustLevelAtLeast('safe-act', 'draft')).toBe(true);
    expect(isTrustLevelAtMost('propose', 'safe-act')).toBe(true);
    expect(isTrustLevelAtLeast('observe', 'propose')).toBe(false);
  });

  it('validates approval request, policy rule, and enforcement outcome contracts', () => {
    const request = ApprovalRequestSchema.parse({
      requestId: 'apr_12345678',
      requestedAt: new Date().toISOString(),
      actor: 'ai-service',
      trustLevel: 'draft',
      action: 'skills.execute',
      classification: {
        id: 'cls_skill_execute',
        category: 'execute',
        impact: 'medium',
        reversible: true,
        requiresNetwork: false,
      },
      audit: {
        traceId: 'trace-approval-1',
        source: 'apps.ai',
        timestamp: new Date().toISOString(),
      },
      context: {
        skillId: 'repo-inspection',
      },
    });

    const rule = ActionPolicyRuleSchema.parse({
      id: 'policy-skill-exec-medium',
      name: 'Skill execution medium impact',
      description: 'Medium-impact execute actions at draft trust require operator approval.',
      actionPattern: 'skills.execute',
      minTrustLevel: 'draft',
      appliesToImpact: ['medium', 'high', 'critical'],
      outcome: 'require_approval',
      requiresApproval: true,
      rationale: 'Execution can produce side effects and must be reviewed at this trust level.',
      enabled: true,
    });

    const decision = ApprovalDecisionRecordSchema.parse({
      decisionId: 'apd_12345678',
      requestId: request.requestId,
      decision: 'requires_operator',
      decidedAt: new Date().toISOString(),
      decidedBy: 'policy-engine',
      reason: 'Operator approval required by policy.',
      policyRuleId: rule.id,
      audit: request.audit,
    });

    const enforcement = ApprovalEnforcementOutcomeSchema.parse({
      request,
      result: {
        requestId: request.requestId,
        outcome: 'require_approval',
        reason: decision.reason,
        policyRuleId: rule.id,
        evaluatedAt: new Date().toISOString(),
        requiresApproval: true,
      },
      decision,
    });

    expect(enforcement.result.outcome).toBe('require_approval');
    expect(enforcement.decision?.decision).toBe('requires_operator');
  });
});
