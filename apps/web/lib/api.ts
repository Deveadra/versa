import type {
  BridgeHealthStatus,
  DomainEvent,
  EnvironmentSummary,
  MemoryRecord,
  SkillMetadata,
  WorkspaceSummary,
} from '@versa/shared';

const CORE_URL = process.env.NEXT_PUBLIC_CORE_URL ?? 'http://localhost:4000';
const AI_URL = process.env.NEXT_PUBLIC_AI_URL ?? 'http://localhost:4100';

async function api<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    cache: 'no-store',
    ...init,
    headers: {
      'content-type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `request failed ${path}`);
  }
  if (response.status === 204) return {} as T;
  return (await response.json()) as T;
}

const coreApi = <T>(path: string, init?: RequestInit) => api<T>(CORE_URL, path, init);
const aiApi = <T>(path: string, init?: RequestInit) => api<T>(AI_URL, path, init);

export const fetchTasks = (scope: 'all' | 'today' | 'overdue' = 'all') =>
  coreApi<{ data: Array<Record<string, string>> }>(`/tasks?scope=${scope}`);

export const createTask = (title: string) =>
  coreApi<{ data: Record<string, string> }>('/tasks', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });

export const completeTask = (taskId: string) =>
  coreApi<{ data: Record<string, string> }>(`/tasks/${taskId}`, {
    method: 'PATCH',
    body: JSON.stringify({ status: 'done', completed_at: new Date().toISOString() }),
  });

export const fetchGoals = () => coreApi<{ data: Array<Record<string, string>> }>('/goals');
export const createGoal = (title: string) =>
  coreApi<{ data: Record<string, string> }>('/goals', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });

export const fetchSchedule = () =>
  coreApi<{ data: Array<Record<string, string>> }>(
    `/schedule?view=day&date=${new Date().toISOString().slice(0, 10)}`,
  );
export const createScheduleBlock = (payload: {
  title: string;
  date: string;
  startTime: string;
  endTime: string;
}) =>
  coreApi<{ data: Record<string, string> }>('/schedule', {
    method: 'POST',
    body: JSON.stringify(payload),
  });

export const fetchPlanner = () => coreApi<{ data: Record<string, unknown> }>('/planner/today');
export const fetchStudyAssignments = () =>
  coreApi<{ data: Array<Record<string, string>> }>('/study/assignments');
export const fetchJobs = () =>
  coreApi<{
    data: { leads: Array<Record<string, string>>; applications: Array<Record<string, string>> };
  }>('/jobs');

export const createJobLead = (company: string, role: string) =>
  coreApi<{ data: Record<string, string> }>('/jobs/leads', {
    method: 'POST',
    body: JSON.stringify({ company, role }),
  });

export type CoreHealth = { ok: boolean };
export type CoreAiHealth = { ok: boolean; fallback?: string };
export type AiHealth = { ok: boolean };
export type AiSkillSummary = { id: string; name: string; metadata: SkillMetadata };

export const fetchCoreHealth = () => coreApi<CoreHealth>('/health');
export const fetchCoreAiHealth = () => coreApi<CoreAiHealth>('/ai/health');
export const fetchEvents = (limit = 25) => coreApi<{ data: DomainEvent[] }>(`/events?limit=${limit}`);
export const fetchWorkspaces = () => coreApi<{ data: WorkspaceSummary[] }>('/workspaces');
export const fetchMemory = (limit = 20) =>
  coreApi<{ data: MemoryRecord[] }>(`/memory?limit=${limit}`);
export const fetchEnvironments = () => coreApi<{ data: EnvironmentSummary[] }>('/environments');

export const fetchAiHealth = () => aiApi<AiHealth>('/health');
export const fetchBridgeHealth = () => aiApi<{ data: BridgeHealthStatus }>('/bridge/health');
export const fetchSkills = () => aiApi<{ data: AiSkillSummary[] }>('/skills');
