import { describe, expect, it } from 'vitest';
import {
  ActionPolicyRuleSchema,
  CapabilityLookupResultSchema,
  CapabilityRegistrationResultSchema,
  CapabilityRegistryEntrySchema,
  BridgeAdapterResultSchema,
  BridgeCapabilitySchema,
  BridgeHealthStatusSchema,
  LegacyBridgeRequestSchema,
  LegacyBridgeResponseSchema,
  AiServiceRequestSchema,
  AiServiceResponseSchema,
  ApprovalDecisionRecordSchema,
  ApprovalEnforcementOutcomeSchema,
  ApprovalRequestSchema,
  EnvironmentContextBundleSchema,
  EnvironmentTwinCreateRequestSchema,
  GatewayHealthStatusSchema,
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

describe('MCP gateway contracts', () => {
  it('validates foundational capability registry and health contracts', () => {
    const entry = CapabilityRegistryEntrySchema.parse({
      capabilityId: 'cap.memory.read',
      kind: 'resource',
      metadata: {
        title: 'Memory read',
        summary: 'Read memory through MCP gateway.',
        owner: 'versa-platform',
        lifecycle: 'active',
        sensitivity: 'internal',
        tags: ['ws09'],
        approvals: {
          required: true,
          policyRef: 'ws08.approvals.default',
          writeAllowed: false,
        },
      },
      resources: [
        {
          id: 'resource.memory.read',
          name: 'memory.read',
          description: 'Read-only memory resource',
          uriTemplate: '/memory?q={text}',
          methods: ['GET'],
          transport: 'http',
        },
      ],
      tools: [],
      prompts: [],
      status: 'active',
    });

    const registration = CapabilityRegistrationResultSchema.parse({
      registered: [entry],
      count: 1,
      status: 'ok',
    });

    const lookup = CapabilityLookupResultSchema.parse({
      capabilityId: entry.capabilityId,
      found: true,
      entry,
    });

    const health = GatewayHealthStatusSchema.parse({
      service: 'mcp-gateway',
      status: 'ok',
      transport: 'http',
      uptimeMs: 1200,
      registeredCapabilities: registration.count,
      telemetryEnabled: true,
      approvalsRequiredByDefault: true,
      timestamp: new Date().toISOString(),
    });

    expect(registration.count).toBe(1);
    expect(lookup.entry?.capabilityId).toBe('cap.memory.read');
    expect(health.service).toBe('mcp-gateway');
  });

  it('preserves additional JSON schema keywords in MCP tool definitions', () => {
    const entry = CapabilityRegistryEntrySchema.parse({
      capabilityId: 'cap.tool.schema-extended',
      kind: 'tool',
      metadata: {
        title: 'Schema extended tool',
        summary: 'Tool with extended JSON schema keywords.',
        owner: 'versa-platform',
        lifecycle: 'active',
        sensitivity: 'internal',
        tags: ['ws09'],
        approvals: {
          required: true,
          policyRef: 'ws08.approvals.default',
          writeAllowed: false,
        },
      },
      resources: [],
      tools: [
        {
          id: 'tool.schema.extended',
          name: 'schema.extended',
          description: 'Tool with JSON schema extensions',
          inputSchema: {
            type: 'object',
            properties: {
              values: {
                type: 'array',
                items: { type: 'string' },
                minItems: 1,
              },
            },
            required: ['values'],
            additionalProperties: false,
            oneOf: [{ required: ['values'] }],
          },
          outputSchema: {
            type: 'object',
            properties: {
              ok: { type: 'boolean', enum: [true] },
            },
            additionalProperties: false,
          },
          sideEffectLevel: 'read',
          approvalsRequired: true,
        },
      ],
      prompts: [],
      status: 'active',
    });

    const tool = entry.tools[0]!;
    expect((tool.inputSchema as Record<string, unknown>).additionalProperties).toBe(false);
    expect((tool.inputSchema as Record<string, unknown>).oneOf).toBeDefined();
    expect((tool.outputSchema as Record<string, unknown>).additionalProperties).toBe(false);
  });
});

describe('AI convergence bridge contracts', () => {
  it('validates bridge health/capabilities/request/response contracts', () => {
    const capability = BridgeCapabilitySchema.parse({
      id: 'legacy.summarize_day',
      name: 'summarize_day',
      description: 'Summarize the operator day using legacy runtime behavior.',
      version: 'legacy-v1',
      owner: 'legacy_python_runtime',
      status: 'available',
      metadata: {
        domain: 'daily-ops',
      },
    });

    const health = BridgeHealthStatusSchema.parse({
      service: 'legacy-python-bridge',
      status: 'ok',
      mode: 'primary',
      targetRuntime: 'legacy_python',
      endpoint: 'http://127.0.0.1:8000',
      latencyMs: 22,
      lastCheckedAt: new Date().toISOString(),
      details: {
        capabilityCount: 1,
      },
    });

    const request = LegacyBridgeRequestSchema.parse({
      requestId: 'breq_12345678',
      operation: 'invoke',
      capabilityId: capability.id,
      payload: {
        text: 'summarize today',
      },
      context: {
        traceId: 'trace-bridge-1',
        actor: 'ai-service',
        source: 'apps.ai',
      },
    });

    const response = LegacyBridgeResponseSchema.parse({
      requestId: request.requestId,
      operation: request.operation,
      status: 'ok',
      targetRuntime: 'legacy_python',
      capabilityId: request.capabilityId,
      data: {
        summary: 'today summary',
      },
    });

    expect(capability.owner).toBe('legacy_python_runtime');
    expect(health.mode).toBe('primary');
    expect(request.operation).toBe('invoke');
    expect(response.status).toBe('ok');
  });

  it('requires capabilityId for invoke bridge operation', () => {
    expect(() =>
      LegacyBridgeRequestSchema.parse({
        requestId: 'breq_missing',
        operation: 'invoke',
        payload: {},
        context: {},
      }),
    ).toThrow('capabilityId is required for invoke operation');
  });

  it('validates AI service request/response and bridge adapter result contracts', () => {
    const request = AiServiceRequestSchema.parse({
      requestId: 'air_12345678',
      capabilityId: 'legacy.summarize_day',
      input: {
        date: '2026-04-27',
      },
      target: 'legacy_python_runtime',
      trace: {
        traceId: 'trace-ai-bridge-1',
      },
    });

    const response = AiServiceResponseSchema.parse({
      requestId: request.requestId,
      capabilityId: request.capabilityId,
      target: request.target,
      status: 'succeeded',
      output: {
        summary: 'daily summary',
      },
    });

    const adapterResult = BridgeAdapterResultSchema.parse({
      bridgeEnabled: true,
      mode: 'primary',
      attempted: true,
      response: {
        requestId: request.requestId,
        operation: 'invoke',
        status: 'ok',
        targetRuntime: 'legacy_python',
        capabilityId: request.capabilityId,
        data: response.output,
      },
    });

    expect(response.target).toBe('legacy_python_runtime');
    expect(adapterResult.attempted).toBe(true);
    expect(adapterResult.response?.status).toBe('ok');
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
