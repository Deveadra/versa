import { fetchTasks } from '../../lib/api';

export default async function TasksPage() {
  try {
    const { data } = await fetchTasks();
    if (data.length === 0) return <p>No tasks yet.</p>;
    return (
      <section>
        <h1>Tasks</h1>
        <ul>
          {data.map((task) => (
            <li key={task.id}>{task.title} ({task.status})</li>
          ))}
        </ul>
      </section>
    );
  } catch {
    return <p>Could not load tasks.</p>;
  }
}
