'use client';

import { FormEvent, useEffect, useState } from 'react';
import { createScheduleBlock, fetchSchedule } from '../../lib/api';

export default function SchedulePage() {
  const [title, setTitle] = useState('');
  const [startTime, setStartTime] = useState('09:00');
  const [endTime, setEndTime] = useState('10:00');
  const [error, setError] = useState('');
  const [blocks, setBlocks] = useState<Array<Record<string, string>>>([]);

  const date = new Date().toISOString().slice(0, 10);
  const load = async () => setBlocks((await fetchSchedule()).data);
  useEffect(() => {
    void load();
  }, []);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    try {
      await createScheduleBlock({ title, date, startTime, endTime });
      setTitle('');
      await load();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <section>
      <h1>Schedule</h1>
      <form
        onSubmit={onSubmit}
        style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}
      >
        <input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Block title"
        />
        <input
          value={startTime}
          onChange={(event) => setStartTime(event.target.value)}
          type="time"
        />
        <input value={endTime} onChange={(event) => setEndTime(event.target.value)} type="time" />
        <button type="submit">Add block</button>
      </form>
      {error ? <p>{error}</p> : null}
      {!blocks.length ? (
        <p>No blocks yet.</p>
      ) : (
        <ul>
          {blocks.map((block) => (
            <li key={block.id}>
              {block.start_time} - {block.end_time} {block.title}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
