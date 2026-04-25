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

export type Task = z.infer<typeof TaskSchema>;
export type DomainEvent = z.infer<typeof DomainEventSchema>;
