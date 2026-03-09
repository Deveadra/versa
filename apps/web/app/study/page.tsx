'use client';

import { useEffect, useState } from 'react';
import { fetchStudyAssignments } from '../../lib/api';

export default function StudyPage() {
  const [assignments, setAssignments] = useState<Array<Record<string, string>>>([]);
  useEffect(() => { fetchStudyAssignments().then((result) => setAssignments(result.data)); }, []);

  return (
    <section>
      <h1>Study</h1>
      {!assignments.length ? <p>No assignments yet.</p> : <ul>{assignments.map((assignment) => <li key={assignment.id}>{assignment.title} ({assignment.status})</li>)}</ul>}
    </section>
  );
}
