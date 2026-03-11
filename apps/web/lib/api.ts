const CORE_URL = process.env.NEXT_PUBLIC_CORE_URL ?? 'http://localhost:4000';

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${CORE_URL}${path}`, {
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

export const fetchTasks = (scope: 'all' | 'today' | 'overdue' = 'all') =>
  api<{ data: Array<Record<string, string>> }>(`/tasks?scope=${scope}`);

export const createTask = (title: string) =>
  api<{ data: Record<string, string> }>('/tasks', { method: 'POST', body: JSON.stringify({ title }) });

export const completeTask = (taskId: string) =>
  api<{ data: Record<string, string> }>(`/tasks/${taskId}`, { method: 'PATCH', body: JSON.stringify({ status: 'done', completed_at: new Date().toISOString() }) });

export const fetchGoals = () => api<{ data: Array<Record<string, string>> }>('/goals');
export const createGoal = (title: string) => api<{ data: Record<string, string> }>('/goals', { method: 'POST', body: JSON.stringify({ title }) });

export const fetchSchedule = () =>
  api<{ data: Array<Record<string, string>> }>(`/schedule?view=day&date=${new Date().toISOString().slice(0, 10)}`);
export const createScheduleBlock = (payload: { title: string; date: string; startTime: string; endTime: string }) =>
  api<{ data: Record<string, string> }>('/schedule', { method: 'POST', body: JSON.stringify(payload) });

export const fetchPlanner = () => api<{ data: Record<string, unknown> }>('/planner/today');
export const fetchStudyAssignments = () => api<{ data: Array<Record<string, string>> }>('/study/assignments');
export const fetchJobs = () => api<{ data: { leads: Array<Record<string, string>>; applications: Array<Record<string, string>> } }>('/jobs');

export const createJobLead = (company: string, role: string) =>
  api<{ data: Record<string, string> }>('/jobs/leads', { method: 'POST', body: JSON.stringify({ company, role }) });
