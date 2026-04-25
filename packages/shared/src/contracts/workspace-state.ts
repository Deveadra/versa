import { z } from 'zod';

export const WorkspaceModeEnum = z.enum(['focus', 'planning', 'review', 'recovery']);

export const WorkspaceStateSchema = z.object({
  workspaceId: z.string().min(3),
  userId: z.string().min(3),
  mode: WorkspaceModeEnum,
  activeTaskIds: z.array(z.string()).default([]),
  activeGoalIds: z.array(z.string()).default([]),
  contextWindowId: z.string().optional(),
  updatedAt: z.string().datetime(),
});

export type WorkspaceState = z.infer<typeof WorkspaceStateSchema>;

