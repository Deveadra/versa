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

export const WorkspaceContextBundleSchema = z.object({
  workspace: WorkspaceRecordSchema,
  summary: WorkspaceSummarySchema,
  recentCheckpoints: z.array(WorkspaceCheckpointSchema).default([]),
});

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

export type Task = z.infer<typeof TaskSchema>;
export type DomainEvent = z.infer<typeof DomainEventSchema>;
export type TraceContext = z.infer<typeof TraceContextSchema>;
export type TelemetryActor = z.infer<typeof TelemetryActorSchema>;
export type TelemetryLevel = z.infer<typeof TelemetryLevelEnum>;
export type TelemetryEvent = z.infer<typeof TelemetryEventSchema>;
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
