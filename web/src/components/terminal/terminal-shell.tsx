import Link from 'next/link';
import {
  Activity,
  CandlestickChart,
  Database,
  Gauge,
  Network,
  ShieldCheck,
  SlidersHorizontal,
} from 'lucide-react';
import type { ReactNode } from 'react';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { href: '/dashboard', label: 'Home', icon: Gauge, help: 'Latest recommendation and key health metrics.' },
  { href: '/signals', label: 'Signal Log', icon: CandlestickChart, help: 'Historical weekly BUY/NO_BUY outputs.' },
  { href: '/explainability', label: 'Why This Signal', icon: ShieldCheck, help: 'Main drivers and validity checks.' },
  { href: '/scenario-lab', label: 'Threshold Tuning', icon: SlidersHorizontal, help: 'Find suitable decision threshold.' },
  { href: '/walk-forward', label: 'Model Health', icon: Network, help: 'Stability across rolling windows.' },
  { href: '/openbb', label: 'Market Data', icon: Database, help: 'OpenBB context and data freshness.' },
] as const;

export function TerminalShell({ pathname, children }: { pathname: string; children: ReactNode }) {
  const activeItem = NAV_ITEMS.find((item) => pathname === item.href || pathname.startsWith(`${item.href}/`));

  return (
    <div className="min-h-screen bg-shell text-foreground">
      <div className="mx-auto grid min-h-screen max-w-[1600px] grid-cols-1 gap-4 px-3 py-3 md:grid-cols-[250px_1fr] lg:px-5">
        <aside className="reveal-up border border-border bg-panel p-3">
          <Link href="/" className="flex items-center gap-3 border border-border bg-panel-strong p-3">
            <div className="grid h-9 w-9 place-content-center border border-accent/50 bg-accent/15 text-accent">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <p className="font-display text-base leading-none">Stock Prediction Interface</p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Weekly model</p>
            </div>
          </Link>

          <nav className="mt-3 space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center justify-between border px-3 py-2 text-sm transition-colors',
                    active
                      ? 'border-accent/60 bg-accent/10 text-accent'
                      : 'border-transparent text-muted hover:border-border hover:bg-panel-strong hover:text-foreground',
                  )}
                >
                  <span className="flex min-w-0 items-start gap-2.5">
                    <Icon className="mt-0.5 h-4 w-4 shrink-0" />
                    <span className="min-w-0">
                      <span className="block">{item.label}</span>
                      <span className="block truncate text-[11px] text-muted">{item.help}</span>
                    </span>
                  </span>
                  <span className={cn('h-1.5 w-1.5 rounded-full bg-transparent', active && 'status-blink bg-accent')} />
                </Link>
              );
            })}
          </nav>

          <div className="mt-3 border border-border bg-panel-strong p-3">
            <div className="flex items-center justify-between">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">System</p>
              <Badge variant="positive">live</Badge>
            </div>
            <div className="mt-2 space-y-1 font-mono text-[11px] text-muted">
              <p>Objective: accuracy</p>
              <p>Refresh: daily 09:00 HKT</p>
              <p>Market: HSI</p>
            </div>
          </div>

          <div className="mt-3 border border-border bg-panel-strong p-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Quick usage</p>
            <ol className="mt-2 space-y-1 text-xs text-slate-200">
              <li>1. Open Home and read latest recommendation.</li>
              <li>2. Check Model Health before acting.</li>
              <li>3. Use Threshold Tuning if behavior drifts.</li>
            </ol>
          </div>
        </aside>

        <main className="space-y-4">
          <header className="reveal-up terminal-sheen relative overflow-hidden border border-border bg-panel p-3">
            <div className="relative z-10 flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Workspace</p>
                <p className="mt-1 font-display text-2xl">{activeItem?.label ?? 'Overview'}</p>
              </div>
              <div className="border border-border bg-panel-strong px-3 py-2 font-mono text-[11px] text-muted">
                Internal HSI analytics tool
              </div>
            </div>
          </header>
          {children}
        </main>
      </div>
    </div>
  );
}
