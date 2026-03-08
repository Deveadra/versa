import { z } from 'zod';

export const DomainEnum = z.enum(['core', 'study', 'jobs', 'health', 'integration', 'system']);
export const StatusEnum = z.enum(['todo', 'in_progress', 'done', 'archived']);
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

export const UserProfileSchema = EntityBase.extend({ displayName: z.string(), email: z.string().email() });
export const TaskSchema = EntityBase.extend({ title: z.string(), description: z.string().optional(), status: StatusEnum, priority: PriorityEnum, projectId: IdSchema.optional() });
export const ProjectSchema = EntityBase.extend({ name: z.string(), status: StatusEnum });
export const GoalSchema = EntityBase.extend({ title: z.string(), status: StatusEnum, targetDate: TimestampSchema.optional() });
export const HabitSchema = EntityBase.extend({ name: z.string(), cadence: z.string() });
export const ScheduleBlockSchema = EntityBase.extend({ startsAt: TimestampSchema, endsAt: TimestampSchema, label: z.string() });
export const CalendarEventSchema = EntityBase.extend({ externalId: z.string(), startsAt: TimestampSchema, endsAt: TimestampSchema });
export const EmailThreadSummarySchema = EntityBase.extend({ subject: z.string(), summary: z.string() });
export const StudyCourseSchema = EntityBase.extend({ name: z.string(), provider: z.string() });
export const StudyAssignmentSchema = EntityBase.extend({ courseId: IdSchema, title: z.string(), dueAt: TimestampSchema.optional() });
export const StudySessionSchema = EntityBase.extend({ courseId: IdSchema, startedAt: TimestampSchema, endedAt: TimestampSchema.optional() });
export const JobLeadSchema = EntityBase.extend({ company: z.string(), role: z.string(), status: StatusEnum });
export const JobApplicationSchema = EntityBase.extend({ company: z.string(), role: z.string(), status: StatusEnum });
export const ResumeAssetSchema = EntityBase.extend({ name: z.string(), version: z.string() });
export const FoodEntrySchema = EntityBase.extend({ mealType: z.string(), calories: z.number().optional() });
export const SymptomEntrySchema = EntityBase.extend({ symptom: z.string(), severity: z.number().min(1).max(10) });
export const WorkTicketSchema = EntityBase.extend({ system: z.string(), title: z.string(), status: StatusEnum });
export const MemoryEventSchema = EntityBase.extend({ summary: z.string(), domain: DomainEnum });
export const ConsentGrantSchema = EntityBase.extend({ scope: z.string(), granted: z.boolean(), grantedAt: TimestampSchema });
export const IntegrationAccountSchema = EntityBase.extend({ provider: z.string(), accountLabel: z.string(), connectedAt: TimestampSchema });
export const SystemEventSchema = EntityBase.extend({ eventType: z.string(), domain: DomainEnum, payload: z.record(z.any()) });

export const EventTypeEnum = z.enum([
  'task.created',
  'task.updated',
  'goal.created',
  'schedule.block.created',
  'study.session.started',
  'study.session.completed',
  'integration.connected',
  'integration.sync.completed',
  'memory.note.recorded',
  'job.application.created',
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
