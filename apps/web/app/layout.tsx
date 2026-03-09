import Link from 'next/link';

const navStyle = { listStyle: 'none', padding: 0, display: 'grid', gap: 8 } as const;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'sans-serif', background: '#0f172a', color: '#e5e7eb', margin: 0 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', minHeight: '100vh' }}>
          <nav style={{ padding: 16, borderRight: '1px solid #334155' }}>
            <h3>Versa</h3>
            <ul style={navStyle}>
              <li><Link href="/today">Command Deck</Link></li>
              <li><Link href="/tasks">Tasks</Link></li>
              <li><Link href="/goals">Goals</Link></li>
              <li><Link href="/schedule">Schedule</Link></li>
              <li><Link href="/study">Study</Link></li>
              <li><Link href="/jobs">Job Hub</Link></li>
              <li><Link href="/settings">Settings</Link></li>
            </ul>
          </nav>
          <main style={{ padding: 24 }}>{children}</main>
        </div>
      </body>
    </html>
  );
}
