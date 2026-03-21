'use client';

import { FormEvent, useEffect, useState } from 'react';
import { createJobLead, fetchJobs } from '../../lib/api';

export default function JobsPage() {
  const [company, setCompany] = useState('');
  const [role, setRole] = useState('');
  const [leads, setLeads] = useState<Array<Record<string, string>>>([]);
  const [apps, setApps] = useState<Array<Record<string, string>>>([]);

  const load = async () => {
    const result = await fetchJobs();
    setLeads(result.data.leads);
    setApps(result.data.applications);
  };
  useEffect(() => { void load(); }, []);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!company || !role) return;
    await createJobLead(company, role);
    setCompany('');
    setRole('');
    await load();
  };

  return (
    <section>
      <h1>Job Hub</h1>
      <form onSubmit={onSubmit} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input value={company} onChange={(event) => setCompany(event.target.value)} placeholder="Company" />
        <input value={role} onChange={(event) => setRole(event.target.value)} placeholder="Role" />
        <button type="submit">Add lead</button>
      </form>
      <h3>Leads</h3>
      {!leads.length ? <p>No leads yet.</p> : <ul>{leads.map((lead) => <li key={lead.id}>{lead.company} — {lead.role}</li>)}</ul>}
      <h3>Applications</h3>
      {!apps.length ? <p>No applications yet.</p> : <ul>{apps.map((app) => <li key={app.id}>{app.status} ({app.lead_id})</li>)}</ul>}
    </section>
  );
}
