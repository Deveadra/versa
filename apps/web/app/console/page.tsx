'use client';

import { useEffect, useMemo, useState } from 'react';
import type { DomainEvent, EnvironmentSummary, MemoryRecord, WorkspaceSummary } from '@versa/shared';
import {
  fetchAiHealth,
  fetchBridgeHealth,
  fetchCoreAiHealth,
  fetchCoreHealth,
  fetchEnvironments,
  fetchEvents,
  fetchMemory,
  fetchSkills,
  fetchWorkspaces,
  type AiSkillSummary,
  type AiHealth,
  type CoreAiHealth,
  type CoreHealth,
} from '../../lib/api';
import { deriveApprovalVisibilitySnapshot } from './model';

const bridgeHealthFallback = {
  data: {
    service: 'ai-bridge',
    status: 'down',
    mode: 'disabled',
    targetRuntime: 'legacy_python',
    lastCheckedAt: new Date(0).toISOString(),
    details: {},
  },
} as const;

type ConsoleData = {
  coreHealth: CoreHealth;
  aiHealth: AiHealth;
  coreAiHealth: CoreAiHealth;
  bridgeHealth: { service: string; status: string; mode: string };
  workspaces: WorkspaceSummary[];
  memory: MemoryRecord[];
  events: DomainEvent[];
  environments: EnvironmentSummary[];
  skills: AiSkillSummary[];
};

const cardStyle = { border: '1px solid #334155', padding: 12, borderRadius: 8 } as const;

export default function OperatorConsolePage() {
  const [data, setData] = useState<ConsoleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [
          coreHealth,
          aiHealth,
          coreAiHealth,
          bridgeHealth,
          workspaces,
          memory,
          events,
          environments,
          skills,
        ] = await Promise.allSettled([
          fetchCoreHealth(),
          fetchAiHealth(),
          fetchCoreAiHealth(),
          fetchBridgeHealth(),
          fetchWorkspaces(),
          fetchMemory(12),
          fetchEvents(20),
          fetchEnvironments(),
          fetchSkills(),
        ]);

        const settledValue = <T,>(
          result: PromiseSettledResult<T>,
          fallback: T,
        ): T => (result.status === 'fulfilled' ? result.value : fallback);

        const bridge = settledValue(bridgeHealth, bridgeHealthFallback);
        const workspaceList = settledValue(workspaces, { data: [] });
        const memoryList = settledValue(memory, { data: [] });
        const eventList = settledValue(events, { data: [] });
        const environmentList = settledValue(environments, { data: [] });
        const skillList = settledValue(skills, { data: [] });

        setData({
          coreHealth: settledValue(coreHealth, { ok: false }),
          aiHealth: settledValue(aiHealth, { ok: false }),
          coreAiHealth: settledValue(coreAiHealth, { ok: false, fallback: 'probe unavailable' }),
          bridgeHealth: {
            service: bridge.data.service,
            status: bridge.data.status,
            mode: bridge.data.mode,
          },
          workspaces: workspaceList.data,
          memory: memoryList.data,
          events: eventList.data,
          environments: environmentList.data,
          skills: skillList.data,
        });
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const approvalView = useMemo(
    () => deriveApprovalVisibilitySnapshot(data?.skills ?? [], data?.events ?? []),
    [data?.events, data?.skills],
  );

  if (loading) return <p>Loading operator console…</p>;
  if (error || !data) return <p>Error loading operator console: {error ?? 'unknown error'}</p>;

  return (
    <section>
      <h1>Operator Console</h1>
      <p>Foundational operational visibility across health, workspaces, memory, traces, approvals, and environment.</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 16 }}>
        <article style={cardStyle}>
          <h3>Health / Status</h3>
          <ul>
            <li>Core API: {data.coreHealth.ok ? 'ok' : 'down'}</li>
            <li>AI service: {data.aiHealth.ok ? 'ok' : 'down'}</li>
            <li>Core→AI probe: {data.coreAiHealth.ok ? 'ok' : 'degraded'}</li>
            <li>
              Bridge: {data.bridgeHealth.status} ({data.bridgeHealth.mode})
            </li>
          </ul>
        </article>

        <article style={cardStyle}>
          <h3>Workspaces</h3>
          <p>{data.workspaces.length} workspace summaries</p>
          <ul>
            {data.workspaces.slice(0, 5).map((workspace) => (
              <li key={workspace.id}>
                {workspace.name} — blockers: {workspace.activeBlockerCount}, next: {workspace.nextActionCount}
              </li>
            ))}
          </ul>
        </article>

        <article style={cardStyle}>
          <h3>Memory summaries</h3>
          <p>{data.memory.length} recent memory records</p>
          <ul>
            {data.memory.slice(0, 5).map((memory) => (
              <li key={memory.id}>
                [{memory.tier}] {memory.summary}
              </li>
            ))}
          </ul>
        </article>

        <article style={cardStyle}>
          <h3>Traces / logs</h3>
          <p>{data.events.length} recent domain events</p>
          <ul>
            {data.events.slice(0, 6).map((event) => (
              <li key={event.eventId}>
                {event.eventType} · {event.domain}
              </li>
            ))}
          </ul>
        </article>

        <article style={cardStyle}>
          <h3>Approvals visibility</h3>
          <ul>
            <li>Governed skills: {approvalView.governedSkillCount}</li>
            <li>Approval-related events in sample: {approvalView.approvalRelatedEventCount}</li>
          </ul>
        </article>

        <article style={cardStyle}>
          <h3>Environment overview</h3>
          <p>{data.environments.length} environment summaries</p>
          <ul>
            {data.environments.slice(0, 5).map((environment) => (
              <li key={environment.id}>
                {environment.name} — records: {environment.recordCount}, procedures: {environment.procedureCount}
              </li>
            ))}
          </ul>
        </article>
      </div>
    </section>
  );
}
