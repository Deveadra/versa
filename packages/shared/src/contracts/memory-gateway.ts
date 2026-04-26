import { z } from 'zod';

export const MemoryNamespaceEnum = z.enum(['profile', 'tasks', 'study', 'jobs', 'health', 'system']);

export const MemoryQuerySchema = z.object({
  namespace: MemoryNamespaceEnum,
  query: z.string().min(1),
  topK: z.number().int().min(1).max(100).default(10),
  minScore: z.number().min(0).max(1).default(0),
});

export const MemoryRecordSchema = z.object({
  memoryId: z.string().min(8),
  namespace: MemoryNamespaceEnum,
  content: z.string().min(1),
  score: z.number().min(0).max(1).optional(),
  tags: z.array(z.string()).default([]),
  createdAt: z.string().datetime(),
});

export type MemoryQuery = z.infer<typeof MemoryQuerySchema>;
export type MemoryRecord = z.infer<typeof MemoryRecordSchema>;

