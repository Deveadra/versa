export async function fetchTasks() {
  const response = await fetch('http://localhost:4000/tasks', { cache: 'no-store' });
  if (!response.ok) throw new Error('failed to fetch tasks');
  return (await response.json()) as { data: Array<{ id: string; title: string; status: string }> };
}
