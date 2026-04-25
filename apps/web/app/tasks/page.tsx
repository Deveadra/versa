'use client';

import { FormEvent, useEffect, useState } from 'react';
import { completeTask, createTask, fetchTasks } from '../../lib/api';

export default function TasksPage() {
  const [title, setTitle] = useState('');
  const [tasks, setTasks] = useState<Array<Record<string, string>>>([]);
  const [scope, setScope] = useState<'all' | 'today' | 'overdue'>('all');

  const load = async (nextScope = scope) => {
    const result = await fetchTasks(nextScope);
    setTasks(result.data);
  };

  useEffect(() => {
    void load(scope);
  }, [scope]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!title.trim()) return;
    await createTask(title.trim());
    setTitle('');
    await load();
  };

  return (
    <section>
      <h1>Tasks</h1>
      <form onSubmit={onSubmit} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Quick add task"
        />
        <button type="submit">Add</button>
      </form>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button onClick={() => setScope('all')}>All</button>
        <button onClick={() => setScope('today')}>Today</button>
        <button onClick={() => setScope('overdue')}>Overdue</button>
      </div>

      {tasks.length === 0 ? (
        <p>No tasks yet.</p>
      ) : (
        <ul>
          {tasks.map((task) => (
            <li key={task.id} style={{ marginBottom: 8 }}>
              <strong>{task.title}</strong> ({task.status}){' '}
              {task.due_date ? `due ${task.due_date}` : ''}
              {task.status !== 'done' ? (
                <button
                  style={{ marginLeft: 8 }}
                  onClick={async () => {
                    await completeTask(String(task.id));
                    await load();
                  }}
                >
                  Complete
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
