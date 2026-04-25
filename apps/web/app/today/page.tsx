'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import {
  fetchGoals,
  fetchPlanner,
  fetchSchedule,
  fetchStudyAssignments,
  fetchTasks,
} from '../../lib/api';

export default function TodayPage() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const [planner, tasks, overdue, schedule, goals, study] = await Promise.all([
          fetchPlanner(),
          fetchTasks('today'),
          fetchTasks('overdue'),
          fetchSchedule(),
          fetchGoals(),
          fetchStudyAssignments(),
        ]);
        setData({
          planner: planner.data,
          todayTasks: tasks.data,
          overdue: overdue.data,
          schedule: schedule.data,
          goals: goals.data,
          study: study.data,
        });
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <p>Loading Command Deck…</p>;
  if (error) return <p>Error loading Command Deck: {error}</p>;

  const planner = (data?.planner ?? {}) as Record<string, unknown>;
  const todayTasks = (data?.todayTasks ?? []) as Array<Record<string, string>>;
  const overdue = (data?.overdue ?? []) as Array<Record<string, string>>;
  const schedule = (data?.schedule ?? []) as Array<Record<string, string>>;
  const goals = (data?.goals ?? []) as Array<Record<string, string>>;
  const study = (data?.study ?? []) as Array<Record<string, string>>;

  return (
    <section>
      <h1>Command Deck</h1>
      <p>
        {time.toLocaleDateString()} {time.toLocaleTimeString()}
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))',
          gap: 16,
        }}
      >
        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Today’s priorities</h3>
          {Array.isArray(planner.priorities) && (planner.priorities as string[]).length > 0 ? (
            <ul>
              {(planner.priorities as string[]).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p>No priorities yet.</p>
          )}
        </article>

        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Upcoming schedule</h3>
          {schedule.length ? (
            <ul>
              {schedule.slice(0, 4).map((block) => (
                <li key={block.id}>
                  {block.start_time}-{block.end_time} {block.title}
                </li>
              ))}
            </ul>
          ) : (
            <p>No blocks planned.</p>
          )}
        </article>

        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Study block</h3>
          <p>
            {study.filter((assignment) => assignment.status !== 'done').length} assignments pending
          </p>
        </article>

        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Goals snapshot</h3>
          <p>{goals.filter((goal) => goal.status === 'active').length} active goals</p>
        </article>
      </div>

      <div
        style={{
          marginTop: 20,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))',
          gap: 16,
        }}
      >
        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Overdue items</h3>
          {overdue.length ? (
            <ul>
              {overdue.map((task) => (
                <li key={task.id}>{task.title}</li>
              ))}
            </ul>
          ) : (
            <p>No overdue tasks.</p>
          )}
        </article>

        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Focus next</h3>
          <p>{String(planner.focusNext ?? 'No suggestion yet')}</p>
          <p>Next block: {(planner.nextBlock as Record<string, string> | null)?.title ?? 'None'}</p>
        </article>

        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Quick add</h3>
          <ul>
            <li>
              <Link href="/tasks">+ Task</Link>
            </li>
            <li>
              <Link href="/goals">+ Goal</Link>
            </li>
            <li>
              <Link href="/schedule">+ Schedule block</Link>
            </li>
            <li>
              <Link href="/jobs">+ Job lead</Link>
            </li>
          </ul>
        </article>

        <article style={{ border: '1px solid #334155', padding: 12 }}>
          <h3>Today tasks</h3>
          {todayTasks.length ? (
            <ul>
              {todayTasks.map((task) => (
                <li key={task.id}>{task.title}</li>
              ))}
            </ul>
          ) : (
            <p>No tasks scheduled for today.</p>
          )}
        </article>
      </div>
    </section>
  );
}
