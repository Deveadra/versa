import Link from 'next/link';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'sans-serif', background: '#0f172a', color: '#e5e7eb', margin: 0 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', minHeight: '100vh' }}>
          <nav style={{ padding: 16, borderRight: '1px solid #334155' }}>
            <h3>Versa</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li><Link href="/today">Today</Link></li>
              <li><Link href="/tasks">Tasks</Link></li>
              <li><Link href="/settings">Settings</Link></li>
            </ul>
          </nav>
          <main style={{ padding: 24 }}>{children}</main>
        </div>
      </body>
    </html>
  );
}
