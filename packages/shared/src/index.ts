import { z } from 'zod';

export const DomainEnum = z.enum(['core', 'study', 'jobs', 'health', 'integration', 'system']);
export const StatusEnum = z.enum([
  'todo',
  'in_progress',
  'done',
  'archived',
  'active',
  'paused',
  'completed',
  'missed',
]);
export const PriorityEnum = z.enum(['low', 'medium', 'high', 'critical']);
export const SourceEnum = z.enum(['manual', 'imported', 'system', 'ai']);
export const SensitivityEnum = z.enum(['public', 'internal', 'private', 'restricted']);

export const IdSchema = z.string().min(8);
export const TimestampSchema = z.string().datetime();

const MetadataSchema = z.object({
  createdAt: TimestampSchema,
  updatedAt: TimestampSchema,
  source: SourceEnum,
  sensitivity: SensitivityEnum,
  traceId: z.string(),
  createdBy: z.string(),
  updatedBy: z.string(),
});

const EntityBase = z.object({ id: IdSchema, metadata: MetadataSchema });

export const UserProfileSchema = EntityBase.extend({
  displayName: z.string(),
  email: z.string().email(),
});
export const TaskSchema = EntityBase.extend({
  title: z.string().min(1),
  description: z.string().optional(),
  status: StatusEnum,
  priority: PriorityEnum,
  dueDate: TimestampSchema.optional(),
  scheduledDate: TimestampSchema.optional(),
  tags: z.array(z.string()).default([]),
  linkedGoalId: IdSchema.optional(),
  domain: DomainEnum.default('core'),
  completedAt: TimestampSchema.optional(),
});
export const ProjectSchema = EntityBase.extend({ name: z.string(), status: StatusEnum });
export const GoalSchema = EntityBase.extend({
  title: z.string(),
  description: z.string().optional(),
  domain: DomainEnum,
  targetType: z.string().optional(),
  targetValue: z.number().optional(),
  currentValue: z.number().optional(),
  deadline: TimestampSchema.optional(),
  status: StatusEnum,
  whyItMatters: z.string().optional(),
});
export const HabitSchema = EntityBase.extend({ name: z.string(), cadence: z.string() });
export const ScheduleBlockSchema = EntityBase.extend({
  title: z.string(),
  type: z.string(),
  startTime: z.string(),
  endTime: z.string(),
  date: z.string(),
  linkedTaskId: IdSchema.optional(),
  linkedGoalId: IdSchema.optional(),
  domain: DomainEnum.optional(),
  notes: z.string().optional(),
  status: StatusEnum,
});
export const CalendarEventSchema = EntityBase.extend({
  externalId: z.string(),
  startsAt: TimestampSchema,
  endsAt: TimestampSchema,
});
export const EmailThreadSummarySchema = EntityBase.extend({
  subject: z.string(),
  summary: z.string(),
});
export const StudyCourseSchema = EntityBase.extend({
  title: z.string(),
  code: z.string().optional(),
  term: z.string().optional(),
  instructor: z.string().optional(),
  status: StatusEnum,
});
export const StudyAssignmentSchema = EntityBase.extend({
  courseId: IdSchema,
  title: z.string(),
  dueDate: TimestampSchema.optional(),
  status: StatusEnum,
});
export const StudySessionSchema = EntityBase.extend({
  courseId: IdSchema,
  assignmentId: IdSchema.optional(),
  startTime: TimestampSchema,
  endTime: TimestampSchema.optional(),
  outcome: z.string().optional(),
});
export const JobLeadSchema = EntityBase.extend({
  company: z.string(),
  role: z.string(),
  source: SourceEnum,
  status: StatusEnum,
});
export const JobApplicationSchema = EntityBase.extend({
  leadId: IdSchema.optional(),
  status: StatusEnum,
  followUpDate: TimestampSchema.optional(),
});
export const ResumeAssetSchema = EntityBase.extend({
  label: z.string(),
  version: z.string().optional(),
  targetRole: z.string().optional(),
});
export const FoodEntrySchema = EntityBase.extend({
  mealType: z.string(),
  calories: z.number().optional(),
});
export const SymptomEntrySchema = EntityBase.extend({
  symptom: z.string(),
  severity: z.number().min(1).max(10),
});
export const WorkTicketSchema = EntityBase.extend({
  system: z.string(),
  title: z.string(),
  status: StatusEnum,
});
export const MemoryEventSchema = EntityBase.extend({ summary: z.string(), domain: DomainEnum });
export const ConsentGrantSchema = EntityBase.extend({
  scope: z.string(),
  granted: z.boolean(),
  grantedAt: TimestampSchema,
});
export const IntegrationAccountSchema = EntityBase.extend({
  provider: z.string(),
  accountLabel: z.string(),
  connectedAt: TimestampSchema,
});
export const SystemEventSchema = EntityBase.extend({
  eventType: z.string(),
  domain: DomainEnum,
  payload: z.record(z.any()),
});

export const EventTypeEnum = z.enum([
  'task.created',
  'task.updated',
  'task.completed',
  'task.archived',
  'goal.created',
  'goal.updated',
  'goal.completed',
  'goal.archived',
  'schedule.block.created',
  'schedule.block.updated',
  'schedule.block.completed',
  'schedule.block.missed',
  'schedule.block.deleted',
  'study.course.created',
  'study.assignment.created',
  'study.assignment.completed',
  'study.session.started',
  'study.session.completed',
  'integration.connected',
  'integration.sync.completed',
  'memory.note.recorded',
  'resume.asset.created',
  'job.lead.created',
  'job.application.created',
  'job.application.updated',
]);

export const DomainEventSchema = z.object({
  eventId: IdSchema,
  eventType: EventTypeEnum,
  actor: z.string(),
  timestamp: TimestampSchema,
  domain: DomainEnum,
  entityRef: z.object({ type: z.string(), id: IdSchema }),
  payload: z.record(z.any()),
  sensitivity: SensitivityEnum,
  traceId: z.string(),
});

export const TelemetryLevelEnum = z.enum(['debug', 'info', 'warn', 'error']);

export const TraceContextSchema = z.object({
  traceId: z.string().min(1),
  correlationId: z.string().min(1).optional(),
  runId: z.string().min(1).optional(),
  requestId: z.string().min(1).optional(),
  parentTraceId: z.string().min(1).optional(),
});

export const TelemetryActorSchema = z.object({
  service: z.string().min(1),
  source: z.string().min(1),
  actorId: z.string().min(1).optional(),
});

export const TelemetryEventSchema = z.object({
  eventId: IdSchema,
  eventType: z.string().min(1),
  level: TelemetryLevelEnum,
  message: z.string().min(1),
  timestamp: TimestampSchema,
  actor: TelemetryActorSchema,
  context: TraceContextSchema,
  attributes: z.record(z.any()).default({}),
});

export const McpTransportEnum = z.enum(['stdio', 'http']);

export const CapabilityKindEnum = z.enum(['resource', 'tool', 'prompt', 'workflow']);

const JsonSchemaShape = z.object({
  type: z.string().min(1),
  properties: z.record(z.any()).default({}),
  required: z.array(z.string().min(1)).optional(),
});

export const McpResourceDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string().min(1),
  uriTemplate: z.string().min(1),
  methods: z.array(z.enum(['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])).min(1),
  transport: McpTransportEnum,
});

export const McpToolDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string().min(1),
  inputSchema: JsonSchemaShape,
  outputSchema: JsonSchemaShape,
  sideEffectLevel: z.enum(['read', 'bounded_write', 'blocked']),
  approvalsRequired: z.boolean().default(true),
});

export const McpPromptDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string().min(1),
  inputSchema: JsonSchemaShape,
  outputSchema: JsonSchemaShape,
  requiresApproval: z.boolean().default(true),
});

export const CapabilityApprovalMetadataSchema = z.object({
  required: z.boolean().default(true),
  policyRef: z.string().min(1),
  writeAllowed: z.boolean().default(false),
});

export const CapabilityMetadataSchema = z.object({
  title: z.string().min(1),
  summary: z.string().min(1),
  owner: z.string().min(1),
  lifecycle: z.enum(['active', 'deprecated', 'experimental']).default('active'),
  sensitivity: SensitivityEnum.default('internal'),
  tags: z.array(z.string().min(1)).default([]),
  approvals: CapabilityApprovalMetadataSchema,
});

export const CapabilityRegistryEntrySchema = z.object({
  capabilityId: z.string().min(1),
  kind: CapabilityKindEnum,
  metadata: CapabilityMetadataSchema,
  resources: z.array(McpResourceDefinitionSchema).default([]),
  tools: z.array(McpToolDefinitionSchema).default([]),
  prompts: z.array(McpPromptDefinitionSchema).default([]),
  status: z.enum(['active', 'disabled']).default('active'),
});

export const GatewayHealthStatusSchema = z.object({
  service: z.string().min(1),
  status: z.enum(['ok', 'degraded', 'down']),
  transport: McpTransportEnum,
  uptimeMs: z.number().int().min(0),
  registeredCapabilities: z.number().int().min(0),
  telemetryEnabled: z.boolean(),
  approvalsRequiredByDefault: z.boolean(),
  timestamp: TimestampSchema,
});

export const CapabilityRegistrationResultSchema = z.object({
  registered: z.array(CapabilityRegistryEntrySchema),
  count: z.number().int().min(0),
  status: z.enum(['ok', 'partial', 'error']).default('ok'),
});

export const CapabilityLookupResultSchema = z.object({
  capabilityId: z.string().min(1),
  found: z.boolean(),
  entry: CapabilityRegistryEntrySchema.optional(),
});

export const DoctrineDecisionPriorityEnum = z.enum([
  'operator_safety',
  'mission_alignment',
  'truthfulness',
  'user_intent',
  'reversibility',
  'execution_speed',
]);

export const DoctrineEscalationSeverityEnum = z.enum(['low', 'medium', 'high', 'critical']);

export const DoctrineResponseStyleSchema = z.object({
  tone: z.string().min(1),
  verbosity: z.enum(['minimal', 'concise', 'detailed']).default('concise'),
  markdownRequired: z.boolean().default(true),
  citationStyle: z.enum(['repo-link', 'none']).default('repo-link'),
  forbiddenPhrases: z.array(z.string()).default([]),
});

export const DoctrineEscalationRuleSchema = z.object({
  id: z.string().min(1),
  condition: z.string().min(1),
  severity: DoctrineEscalationSeverityEnum,
  action: z.string().min(1),
});

export const DoctrineAutonomyBoundarySchema = z.object({
  action: z.string().min(1),
  requiresApproval: z.boolean(),
  rationale: z.string().min(1),
});

export const DoctrineSafetyRuleSchema = z.object({
  id: z.string().min(1),
  rule: z.string().min(1),
  rationale: z.string().min(1),
});

export const DoctrineSchema = z.object({
  doctrineId: z.string().min(1),
  version: z.string().min(1),
  mission: z.string().min(1),
  operatorPrinciples: z.array(z.string().min(1)).min(1),
  responseStyle: DoctrineResponseStyleSchema,
  decisionPriorities: z.array(DoctrineDecisionPriorityEnum).min(1),
  escalationRules: z.array(DoctrineEscalationRuleSchema).default([]),
  autonomyBoundaries: z.array(DoctrineAutonomyBoundarySchema).default([]),
  safetyNoGoActions: z.array(DoctrineSafetyRuleSchema).min(1),
  ownership: z.object({
    team: z.string().min(1),
    maintainers: z.array(z.string().min(1)).min(1),
  }),
  metadata: z.object({
    createdAt: TimestampSchema,
    updatedAt: TimestampSchema,
    changeSummary: z.string().optional(),
  }),
});

export const WorkspaceBlockerStatusEnum = z.enum(['active', 'mitigated', 'resolved']);

export const WorkspaceBlockerSchema = z.object({
  description: z.string().min(1),
  status: WorkspaceBlockerStatusEnum.default('active'),
  owner: z.string().min(1).optional(),
  notes: z.string().optional(),
});

export const WorkspaceDecisionSchema = z.object({
  summary: z.string().min(1),
  rationale: z.string().optional(),
  decidedAt: TimestampSchema,
});

export const WorkspaceFileReferenceSchema = z.object({
  path: z.string().min(1),
  reason: z.string().optional(),
});

export const WorkspaceCommandReferenceSchema = z.object({
  command: z.string().min(1),
  description: z.string().optional(),
});

export const WorkspaceProcedureSchema = z.object({
  name: z.string().min(1),
  command: z.string().min(1),
  validatedAt: TimestampSchema.optional(),
  notes: z.string().optional(),
});

export const WorkspaceRecommendedActionSchema = z.object({
  action: z.string().min(1),
  priority: PriorityEnum.default('medium'),
  rationale: z.string().optional(),
});

export const WorkspaceStateSchema = z.object({
  currentObjective: z.string().min(1),
  activeBlockers: z.array(WorkspaceBlockerSchema).default([]),
  recentDecisions: z.array(WorkspaceDecisionSchema).default([]),
  importantFiles: z.array(WorkspaceFileReferenceSchema).default([]),
  knownCommands: z.array(WorkspaceCommandReferenceSchema).default([]),
  validatedProcedures: z.array(WorkspaceProcedureSchema).default([]),
  nextRecommendedActions: z.array(WorkspaceRecommendedActionSchema).default([]),
  updatedAt: TimestampSchema,
});

export const WorkspaceStatePatchSchema = z
  .object({
    currentObjective: z.string().min(1).optional(),
    activeBlockers: z.array(WorkspaceBlockerSchema).optional(),
    recentDecisions: z.array(WorkspaceDecisionSchema).optional(),
    importantFiles: z.array(WorkspaceFileReferenceSchema).optional(),
    knownCommands: z.array(WorkspaceCommandReferenceSchema).optional(),
    validatedProcedures: z.array(WorkspaceProcedureSchema).optional(),
    nextRecommendedActions: z.array(WorkspaceRecommendedActionSchema).optional(),
  })
  .refine((input) => Object.values(input).some((value) => value !== undefined), {
    message: 'at least one workspace state field must be provided',
  });

export const WorkspaceMetadataSchema = z.object({
  owner: z.string().min(1).optional(),
  tags: z.array(z.string().min(1)).default([]),
  source: SourceEnum.default('manual'),
  createdAt: TimestampSchema,
  updatedAt: TimestampSchema,
  lastActivatedAt: TimestampSchema.optional(),
});

export const WorkspaceIdentitySchema = z.object({
  id: IdSchema,
  slug: z.string().min(2).regex(/^[a-z0-9][a-z0-9._/-]*$/),
  name: z.string().min(1),
  repository: z.string().min(1).optional(),
});

export const WorkspaceRecordSchema = WorkspaceIdentitySchema.extend({
  metadata: WorkspaceMetadataSchema,
  state: WorkspaceStateSchema,
});

export const WorkspaceCreateRequestSchema = z.object({
  slug: WorkspaceIdentitySchema.shape.slug,
  name: WorkspaceIdentitySchema.shape.name,
  repository: WorkspaceIdentitySchema.shape.repository,
  metadata: z
    .object({
      owner: WorkspaceMetadataSchema.shape.owner,
      tags: WorkspaceMetadataSchema.shape.tags,
      source: WorkspaceMetadataSchema.shape.source,
      lastActivatedAt: WorkspaceMetadataSchema.shape.lastActivatedAt,
    })
    .default({}),
  state: WorkspaceStateSchema.omit({ updatedAt: true }),
});

export const WorkspaceCheckpointSchema = z.object({
  id: IdSchema,
  workspaceId: IdSchema,
  summary: z.string().min(1),
  snapshot: WorkspaceStateSchema,
  createdAt: TimestampSchema,
  createdBy: z.string().min(1),
});

export const WorkspaceCheckpointCreateRequestSchema = z.object({
  summary: z.string().min(1),
  createdBy: z.string().min(1),
  snapshot: WorkspaceStateSchema.omit({ updatedAt: true }).optional(),
});

export const WorkspaceSummarySchema = z.object({
  id: IdSchema,
  slug: WorkspaceIdentitySchema.shape.slug,
  name: WorkspaceIdentitySchema.shape.name,
  currentObjective: z.string().min(1),
  activeBlockerCount: z.number().int().min(0),
  nextActionCount: z.number().int().min(0),
  updatedAt: TimestampSchema,
  lastActivatedAt: TimestampSchema.optional(),
});

export const normalizeWorkspaceCheckpointLimit = (limit?: number, fallback = 10) => {
  const normalizedFallback = Number.isFinite(fallback) ? Math.max(1, Math.floor(fallback)) : 10;
  if (limit === undefined) return normalizedFallback;
  if (!Number.isFinite(limit)) return normalizedFallback;
  return Math.max(1, Math.floor(limit));
};

export const deriveWorkspaceSummary = (workspace: WorkspaceRecord): WorkspaceSummary =>
  WorkspaceSummarySchema.parse({
    id: workspace.id,
    slug: workspace.slug,
    name: workspace.name,
    currentObjective: workspace.state.currentObjective,
    activeBlockerCount: workspace.state.activeBlockers.filter((b) => b.status === 'active').length,
    nextActionCount: workspace.state.nextRecommendedActions.length,
    updatedAt: workspace.metadata.updatedAt,
    lastActivatedAt: workspace.metadata.lastActivatedAt,
  });

export const WorkspaceContextBundleSchema = z.object({
  workspace: WorkspaceRecordSchema,
  summary: WorkspaceSummarySchema,
  recentCheckpoints: z.array(WorkspaceCheckpointSchema).default([]),
});

export const EnvironmentEntityKindEnum = z.enum([
  'machine',
  'service',
  'dashboard',
  'repository',
  'access_path',
  'command',
  'procedure',
  'environment',
]);

export const EnvironmentRecordSchema = z.object({
  id: IdSchema,
  environmentId: IdSchema,
  kind: EnvironmentEntityKindEnum,
  name: z.string().min(1),
  description: z.string().optional(),
  attributes: z.record(z.any()).default({}),
  metadata: z.object({
    source: SourceEnum.default('manual'),
    tags: z.array(z.string().min(1)).default([]),
    confidence: z.number().min(0).max(1).default(0.8),
    createdAt: TimestampSchema,
    updatedAt: TimestampSchema,
    validatedAt: TimestampSchema.optional(),
  }),
});

export const EnvironmentRelationshipSchema = z.object({
  id: IdSchema,
  environmentId: IdSchema,
  fromEntityId: IdSchema,
  toEntityId: IdSchema,
  relation: z.enum([
    'hosts',
    'depends_on',
    'owned_by',
    'observed_in',
    'accessed_via',
    'documents',
    'mirrors',
    'runs_on',
  ]),
  direction: z.enum(['directed', 'bidirectional']).default('directed'),
  notes: z.string().optional(),
  createdAt: TimestampSchema,
});

export const EnvironmentAccessPathSchema = z.object({
  id: IdSchema,
  environmentId: IdSchema,
  entityId: IdSchema,
  name: z.string().min(1),
  method: z.enum(['ssh', 'http', 'https', 'cli', 'vpn', 'rdp', 'other']),
  endpoint: z.string().min(1),
  prerequisites: z.array(z.string().min(1)).default([]),
  commandRefIds: z.array(IdSchema).default([]),
  notes: z.string().optional(),
  createdAt: TimestampSchema,
  validatedAt: TimestampSchema.optional(),
});

export const EnvironmentProcedureStepSchema = z.object({
  order: z.number().int().min(1),
  instruction: z.string().min(1),
  commandRefId: IdSchema.optional(),
  expectedOutcome: z.string().optional(),
  onFailure: z.string().optional(),
});

export const EnvironmentProcedureSchema = z.object({
  id: IdSchema,
  environmentId: IdSchema,
  name: z.string().min(1),
  intent: z.string().min(1),
  targetEntityIds: z.array(IdSchema).default([]),
  steps: z.array(EnvironmentProcedureStepSchema).min(1),
  lastValidatedAt: TimestampSchema.optional(),
  owner: z.string().min(1).optional(),
  tags: z.array(z.string().min(1)).default([]),
  createdAt: TimestampSchema,
  updatedAt: TimestampSchema,
});

export const EnvironmentMetadataSchema = z.object({
  owner: z.string().min(1).optional(),
  tags: z.array(z.string().min(1)).default([]),
  source: SourceEnum.default('manual'),
  createdAt: TimestampSchema,
  updatedAt: TimestampSchema,
  lastValidatedAt: TimestampSchema.optional(),
});

export const EnvironmentTwinRecordSchema = z.object({
  id: IdSchema,
  slug: z.string().min(2).regex(/^[a-z0-9][a-z0-9._/-]*$/),
  name: z.string().min(1),
  metadata: EnvironmentMetadataSchema,
});

export const EnvironmentTwinCreateRequestSchema = z.object({
  slug: EnvironmentTwinRecordSchema.shape.slug,
  name: EnvironmentTwinRecordSchema.shape.name,
  metadata: z
    .object({
      owner: EnvironmentMetadataSchema.shape.owner,
      tags: EnvironmentMetadataSchema.shape.tags,
      source: EnvironmentMetadataSchema.shape.source,
      lastValidatedAt: EnvironmentMetadataSchema.shape.lastValidatedAt,
    })
    .default({}),
});

export const EnvironmentSummarySchema = z.object({
  id: IdSchema,
  slug: EnvironmentTwinRecordSchema.shape.slug,
  name: EnvironmentTwinRecordSchema.shape.name,
  owner: z.string().optional(),
  recordCount: z.number().int().min(0),
  relationshipCount: z.number().int().min(0),
  procedureCount: z.number().int().min(0),
  updatedAt: TimestampSchema,
  lastValidatedAt: TimestampSchema.optional(),
});

export const EnvironmentContextBundleSchema = z.object({
  environment: EnvironmentTwinRecordSchema,
  records: z.array(EnvironmentRecordSchema).default([]),
  relationships: z.array(EnvironmentRelationshipSchema).default([]),
  accessPaths: z.array(EnvironmentAccessPathSchema).default([]),
  procedures: z.array(EnvironmentProcedureSchema).default([]),
});

export const normalizeEnvironmentItemLimit = (
  limit?: number,
  fallback = 50,
  max = 200,
) => {
  const normalizedFallback = Number.isFinite(fallback) ? Math.max(1, Math.floor(fallback)) : 50;
  const normalizedMax = Number.isFinite(max)
    ? Math.max(normalizedFallback, Math.floor(max))
    : normalizedFallback;
  if (limit === undefined) return normalizedFallback;
  if (!Number.isFinite(limit)) return normalizedFallback;
  return Math.min(normalizedMax, Math.max(1, Math.floor(limit)));
};

export const MemoryTierEnum = z.enum(['session', 'episodic', 'semantic', 'procedural']);
export const MemoryRetentionStrategyEnum = z.enum(['session', 'ttl', 'durable']);

export const MemoryProvenanceSchema = z.object({
  actor: z.string().min(1),
  traceId: z.string().min(1).optional(),
  eventId: z.string().min(1).optional(),
  subsystem: z.string().min(1).optional(),
  sourceMemoryIds: z.array(IdSchema).optional(),
  notes: z.string().optional(),
});

export const MemoryRetentionSchema = z.object({
  strategy: MemoryRetentionStrategyEnum,
  ttlDays: z.number().int().positive().optional(),
  decayRate: z.number().min(0).max(1).optional(),
});

export const MemoryMetadataSchema = z.object({
  confidence: z.number().min(0).max(1),
  source: SourceEnum,
  sensitivity: SensitivityEnum,
  retention: MemoryRetentionSchema,
  provenance: MemoryProvenanceSchema,
  tags: z.array(z.string().min(1)).default([]),
});

export const MemoryRecordSchema = z.object({
  id: IdSchema,
  tier: MemoryTierEnum,
  summary: z.string().min(1),
  content: z.record(z.any()).default({}),
  metadata: MemoryMetadataSchema,
  createdAt: TimestampSchema,
  updatedAt: TimestampSchema,
  lastAccessedAt: TimestampSchema.optional(),
});

export const MemoryWriteRequestSchema = z.object({
  tier: MemoryTierEnum,
  summary: z.string().min(1),
  content: z.record(z.any()).default({}),
  metadata: MemoryMetadataSchema,
});

export const MemoryReadRequestSchema = z.object({
  text: z.string().min(1).optional(),
  tiers: z.array(MemoryTierEnum).min(1).optional(),
  minConfidence: z.number().min(0).max(1).optional(),
  limit: z.number().int().min(1).max(100).default(20),
});

export const MemoryConsolidationRequestSchema = z.object({
  sourceMemoryIds: z.array(IdSchema).min(2),
  targetTier: z.enum(['semantic', 'procedural']),
  summary: z.string().min(1),
  reason: z.string().min(1),
  content: z.record(z.any()).default({}),
  metadata: MemoryMetadataSchema,
});

export const MemoryConsolidationResultSchema = z.object({
  promotedMemory: MemoryRecordSchema,
  linkedSourceCount: z.number().int().min(0),
});

export const SkillExecutionStatusEnum = z.enum([
  'succeeded',
  'failed',
  'blocked',
  'invalid_request',
]);

export const SkillInputFieldSchema = z.object({
  name: z.string().min(1),
  description: z.string().min(1),
  required: z.boolean().default(true),
  schemaHint: z.string().min(1).optional(),
});

export const SkillOutputFieldSchema = z.object({
  name: z.string().min(1),
  description: z.string().min(1),
  schemaHint: z.string().min(1).optional(),
});

export const SkillValidationRequirementSchema = z.object({
  id: z.string().min(1),
  description: z.string().min(1),
  required: z.boolean().default(true),
});

export const SkillFailureHandlingSchema = z.object({
  retryable: z.boolean().default(false),
  maxRetries: z.number().int().min(0).default(0),
  escalation: z.string().min(1).optional(),
});

export const SkillApprovalRequirementSchema = z.object({
  required: z.boolean().default(false),
  rationale: z.string().min(1).optional(),
  policyRef: z.string().min(1).optional(),
});

export const SkillMetadataSchema = z.object({
  description: z.string().min(1),
  version: z.string().min(1),
  inputs: z.array(SkillInputFieldSchema).default([]),
  outputs: z.array(SkillOutputFieldSchema).default([]),
  requiredTools: z.array(z.string().min(1)).default([]),
  requiredResources: z.array(z.string().min(1)).default([]),
  validationChecks: z.array(SkillValidationRequirementSchema).default([]),
  failureHandling: SkillFailureHandlingSchema,
  approval: SkillApprovalRequirementSchema,
});

export const SkillDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  metadata: SkillMetadataSchema,
  tags: z.array(z.string().min(1)).default([]),
  bounded: z.boolean().default(true),
  deterministic: z.boolean().default(true),
});

export const SkillExecutionContextSchema = z.object({
  traceId: z.string().min(1).optional(),
  actor: z.string().min(1).optional(),
  workspace: z.string().min(1).optional(),
});

export const SkillExecutionRequestSchema = z
  .object({
    skillId: z.string().min(1).optional(),
    skillName: z.string().min(1).optional(),
    input: z.record(z.any()).default({}),
    context: SkillExecutionContextSchema.default({}),
  })
  .refine((value) => Boolean(value.skillId || value.skillName), {
    message: 'skillId or skillName is required',
  });

export const SkillValidationResultSchema = z.object({
  id: z.string().min(1),
  passed: z.boolean(),
  message: z.string().optional(),
});

export const SkillErrorSchema = z.object({
  code: z.string().min(1),
  message: z.string().min(1),
  details: z.record(z.any()).optional(),
});

export const SkillExecutionResultSchema = z.object({
  executionId: z.string().min(1),
  skillId: z.string().min(1),
  skillName: z.string().min(1),
  status: SkillExecutionStatusEnum,
  startedAt: TimestampSchema,
  completedAt: TimestampSchema,
  output: z.record(z.any()).default({}),
  validation: z.object({
    passed: z.boolean(),
    checks: z.array(SkillValidationResultSchema).default([]),
  }),
  error: SkillErrorSchema.optional(),
});

export const TrustLevelEnum = z.enum([
  'observe',
  'propose',
  'draft',
  'safe-act',
  'bounded-autonomous',
]);

export const ActionImpactEnum = z.enum(['low', 'medium', 'high', 'critical']);

export const ActionClassificationSchema = z.object({
  id: z.string().min(1),
  category: z.enum(['read', 'write', 'execute', 'integration', 'system']),
  impact: ActionImpactEnum,
  reversible: z.boolean(),
  requiresNetwork: z.boolean().default(false),
  description: z.string().optional(),
});

export const ApprovalAuditMetadataSchema = z.object({
  traceId: z.string().min(1),
  correlationId: z.string().min(1).optional(),
  requestId: z.string().min(1).optional(),
  source: z.string().min(1),
  timestamp: TimestampSchema,
});

export const ApprovalRequestSchema = z.object({
  requestId: z.string().min(1),
  requestedAt: TimestampSchema,
  actor: z.string().min(1),
  trustLevel: TrustLevelEnum,
  action: z.string().min(1),
  classification: ActionClassificationSchema,
  audit: ApprovalAuditMetadataSchema,
  context: z.record(z.any()).default({}),
});

export const ApprovalDecisionEnum = z.enum([
  'approved',
  'denied',
  'auto_approved',
  'requires_operator',
]);

export const ApprovalDecisionRecordSchema = z.object({
  decisionId: z.string().min(1),
  requestId: z.string().min(1),
  decision: ApprovalDecisionEnum,
  decidedAt: TimestampSchema,
  decidedBy: z.string().min(1),
  reason: z.string().min(1),
  policyRuleId: z.string().min(1).optional(),
  audit: ApprovalAuditMetadataSchema,
});

export const ApprovalOutcomeEnum = z.enum(['allow', 'require_approval', 'deny']);

export const ActionPolicyRuleSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string().min(1),
  actionPattern: z.string().min(1).default('*'),
  minTrustLevel: TrustLevelEnum.default('observe'),
  maxTrustLevel: TrustLevelEnum.optional(),
  appliesToImpact: z.array(ActionImpactEnum).default([]),
  outcome: ApprovalOutcomeEnum,
  requiresApproval: z.boolean().default(false),
  rationale: z.string().min(1),
  enabled: z.boolean().default(true),
});

export const ApprovalResultSchema = z.object({
  requestId: z.string().min(1),
  outcome: ApprovalOutcomeEnum,
  reason: z.string().min(1),
  policyRuleId: z.string().min(1).optional(),
  evaluatedAt: TimestampSchema,
  requiresApproval: z.boolean().default(false),
});

export const ApprovalEnforcementOutcomeSchema = z.object({
  request: ApprovalRequestSchema,
  result: ApprovalResultSchema,
  decision: ApprovalDecisionRecordSchema.optional(),
});

export type Task = z.infer<typeof TaskSchema>;
export type DomainEvent = z.infer<typeof DomainEventSchema>;
export type TraceContext = z.infer<typeof TraceContextSchema>;
export type TelemetryActor = z.infer<typeof TelemetryActorSchema>;
export type TelemetryLevel = z.infer<typeof TelemetryLevelEnum>;
export type TelemetryEvent = z.infer<typeof TelemetryEventSchema>;
export type McpTransport = z.infer<typeof McpTransportEnum>;
export type CapabilityKind = z.infer<typeof CapabilityKindEnum>;
export type McpResourceDefinition = z.infer<typeof McpResourceDefinitionSchema>;
export type McpToolDefinition = z.infer<typeof McpToolDefinitionSchema>;
export type McpPromptDefinition = z.infer<typeof McpPromptDefinitionSchema>;
export type CapabilityApprovalMetadata = z.infer<typeof CapabilityApprovalMetadataSchema>;
export type CapabilityMetadata = z.infer<typeof CapabilityMetadataSchema>;
export type CapabilityRegistryEntry = z.infer<typeof CapabilityRegistryEntrySchema>;
export type GatewayHealthStatus = z.infer<typeof GatewayHealthStatusSchema>;
export type CapabilityRegistrationResult = z.infer<typeof CapabilityRegistrationResultSchema>;
export type CapabilityLookupResult = z.infer<typeof CapabilityLookupResultSchema>;
export type DoctrineDecisionPriority = z.infer<typeof DoctrineDecisionPriorityEnum>;
export type DoctrineEscalationSeverity = z.infer<typeof DoctrineEscalationSeverityEnum>;
export type DoctrineResponseStyle = z.infer<typeof DoctrineResponseStyleSchema>;
export type DoctrineEscalationRule = z.infer<typeof DoctrineEscalationRuleSchema>;
export type DoctrineAutonomyBoundary = z.infer<typeof DoctrineAutonomyBoundarySchema>;
export type DoctrineSafetyRule = z.infer<typeof DoctrineSafetyRuleSchema>;
export type Doctrine = z.infer<typeof DoctrineSchema>;
export type WorkspaceBlockerStatus = z.infer<typeof WorkspaceBlockerStatusEnum>;
export type WorkspaceBlocker = z.infer<typeof WorkspaceBlockerSchema>;
export type WorkspaceDecision = z.infer<typeof WorkspaceDecisionSchema>;
export type WorkspaceFileReference = z.infer<typeof WorkspaceFileReferenceSchema>;
export type WorkspaceCommandReference = z.infer<typeof WorkspaceCommandReferenceSchema>;
export type WorkspaceProcedure = z.infer<typeof WorkspaceProcedureSchema>;
export type WorkspaceRecommendedAction = z.infer<typeof WorkspaceRecommendedActionSchema>;
export type WorkspaceState = z.infer<typeof WorkspaceStateSchema>;
export type WorkspaceStatePatch = z.infer<typeof WorkspaceStatePatchSchema>;
export type WorkspaceMetadata = z.infer<typeof WorkspaceMetadataSchema>;
export type WorkspaceIdentity = z.infer<typeof WorkspaceIdentitySchema>;
export type WorkspaceRecord = z.infer<typeof WorkspaceRecordSchema>;
export type WorkspaceCreateRequest = z.infer<typeof WorkspaceCreateRequestSchema>;
export type WorkspaceCheckpoint = z.infer<typeof WorkspaceCheckpointSchema>;
export type WorkspaceCheckpointCreateRequest = z.infer<
  typeof WorkspaceCheckpointCreateRequestSchema
>;
export type WorkspaceSummary = z.infer<typeof WorkspaceSummarySchema>;
export type WorkspaceContextBundle = z.infer<typeof WorkspaceContextBundleSchema>;
export type EnvironmentEntityKind = z.infer<typeof EnvironmentEntityKindEnum>;
export type EnvironmentRecord = z.infer<typeof EnvironmentRecordSchema>;
export type EnvironmentRelationship = z.infer<typeof EnvironmentRelationshipSchema>;
export type EnvironmentAccessPath = z.infer<typeof EnvironmentAccessPathSchema>;
export type EnvironmentProcedureStep = z.infer<typeof EnvironmentProcedureStepSchema>;
export type EnvironmentProcedure = z.infer<typeof EnvironmentProcedureSchema>;
export type EnvironmentMetadata = z.infer<typeof EnvironmentMetadataSchema>;
export type EnvironmentTwinRecord = z.infer<typeof EnvironmentTwinRecordSchema>;
export type EnvironmentTwinCreateRequest = z.infer<typeof EnvironmentTwinCreateRequestSchema>;
export type EnvironmentSummary = z.infer<typeof EnvironmentSummarySchema>;
export type EnvironmentContextBundle = z.infer<typeof EnvironmentContextBundleSchema>;
export type MemoryTier = z.infer<typeof MemoryTierEnum>;
export type MemoryRetentionStrategy = z.infer<typeof MemoryRetentionStrategyEnum>;
export type MemoryProvenance = z.infer<typeof MemoryProvenanceSchema>;
export type MemoryRetention = z.infer<typeof MemoryRetentionSchema>;
export type MemoryMetadata = z.infer<typeof MemoryMetadataSchema>;
export type MemoryRecord = z.infer<typeof MemoryRecordSchema>;
export type MemoryWriteRequest = z.infer<typeof MemoryWriteRequestSchema>;
export type MemoryReadRequest = z.infer<typeof MemoryReadRequestSchema>;
export type MemoryConsolidationRequest = z.infer<typeof MemoryConsolidationRequestSchema>;
export type MemoryConsolidationResult = z.infer<typeof MemoryConsolidationResultSchema>;
export type SkillExecutionStatus = z.infer<typeof SkillExecutionStatusEnum>;
export type SkillInputField = z.infer<typeof SkillInputFieldSchema>;
export type SkillOutputField = z.infer<typeof SkillOutputFieldSchema>;
export type SkillValidationRequirement = z.infer<typeof SkillValidationRequirementSchema>;
export type SkillFailureHandling = z.infer<typeof SkillFailureHandlingSchema>;
export type SkillApprovalRequirement = z.infer<typeof SkillApprovalRequirementSchema>;
export type SkillMetadata = z.infer<typeof SkillMetadataSchema>;
export type SkillDefinition = z.infer<typeof SkillDefinitionSchema>;
export type SkillExecutionContext = z.infer<typeof SkillExecutionContextSchema>;
export type SkillExecutionRequest = z.infer<typeof SkillExecutionRequestSchema>;
export type SkillValidationResult = z.infer<typeof SkillValidationResultSchema>;
export type SkillError = z.infer<typeof SkillErrorSchema>;
export type SkillExecutionResult = z.infer<typeof SkillExecutionResultSchema>;
export type TrustLevel = z.infer<typeof TrustLevelEnum>;
export type ActionImpact = z.infer<typeof ActionImpactEnum>;
export type ActionClassification = z.infer<typeof ActionClassificationSchema>;
export type ApprovalAuditMetadata = z.infer<typeof ApprovalAuditMetadataSchema>;
export type ApprovalRequest = z.infer<typeof ApprovalRequestSchema>;
export type ApprovalDecision = z.infer<typeof ApprovalDecisionEnum>;
export type ApprovalDecisionRecord = z.infer<typeof ApprovalDecisionRecordSchema>;
export type ApprovalOutcome = z.infer<typeof ApprovalOutcomeEnum>;
export type ActionPolicyRule = z.infer<typeof ActionPolicyRuleSchema>;
export type ApprovalResult = z.infer<typeof ApprovalResultSchema>;
export type ApprovalEnforcementOutcome = z.infer<typeof ApprovalEnforcementOutcomeSchema>;

const TrustLevelOrder: TrustLevel[] = [
  'observe',
  'propose',
  'draft',
  'safe-act',
  'bounded-autonomous',
];

export const trustLevelRank = (level: TrustLevel): number => TrustLevelOrder.indexOf(level);

export const isTrustLevelAtLeast = (level: TrustLevel, minimum: TrustLevel): boolean =>
  trustLevelRank(level) >= trustLevelRank(minimum);

export const isTrustLevelAtMost = (level: TrustLevel, maximum: TrustLevel): boolean =>
  trustLevelRank(level) <= trustLevelRank(maximum);
