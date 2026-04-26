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
