'use client';

import { FormEvent, useEffect, useState } from 'react';
import { createGoal, fetchGoals } from '../../lib/api';

export default function GoalsPage() {
  const [title, setTitle] = useState('');
  const [goals, setGoals] = useState<Array<Record<string, string>>>([]);

  const load = async () => setGoals((await fetchGoals()).data);
  useEffect(() => { void load(); }, []);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!title.trim()) return;
    await createGoal(title.trim());
    setTitle('');
    await load();
  };

  return (
    <section>
      <h1>Goals</h1>
      <form onSubmit={onSubmit} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Create goal" />
        <button type="submit">Add</button>
      </form>
      {!goals.length ? <p>No goals yet.</p> : <ul>{goals.map((goal) => <li key={goal.id}>{goal.title} ({goal.status})</li>)}</ul>}
    </section>
  );
}
