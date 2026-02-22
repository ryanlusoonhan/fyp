import Link from 'next/link';
import { Activity, CandlestickChart, CreditCard, Gauge, Network, SlidersHorizontal, ShieldCheck } from 'lucide-react';
import type { ReactNode } from 'react';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { href: '/dashboard', label: 'Command', icon: Gauge },
  { href: '/signals', label: 'Signals', icon: CandlestickChart },
  { href: '/explainability', label: 'Explain', icon: ShieldCheck },
  { href: '/scenario-lab', label: 'Scenario', icon: SlidersHorizontal },
  { href: '/walk-forward', label: 'Walk-Forward', icon: Network },
  { href: '/pricing', label: 'Pricing', icon: CreditCard },
] as const;

export function TerminalShell({ pathname, children }: { pathname: string; children: ReactNode }) {
  const activeItem = NAV_ITEMS.find((item) => pathname === item.href || pathname.startsWith(`${item.href}/`));

  return (
    <div className="min-h-screen bg-shell text-foreground">
      <div className="glass-grid absolute inset-0 -z-10" />
      <div className="mx-auto grid min-h-screen max-w-[1520px] grid-cols-1 gap-4 px-4 py-4 md:grid-cols-[240px_1fr] lg:px-6">
        <aside className="reveal-up rounded-3xl border border-border/70 bg-panel p-4 shadow-[0_20px_42px_rgba(4,6,14,0.45)]">
          <Link href="/" className="flex items-center gap-3 rounded-2xl border border-border bg-panel-strong p-3">
            <div className="grid h-10 w-10 place-content-center rounded-xl bg-[linear-gradient(140deg,#f5a524,#ffbe55)] text-accent-foreground">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <p className="font-display text-base">Nell Terminal</p>
              <p className="text-[11px] uppercase tracking-[0.16em] text-muted">Weekly HSI</p>
            </div>
          </Link>

          <nav className="mt-4 space-y-2">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'group flex items-center justify-between rounded-xl border px-3 py-2.5 text-sm transition',
                    active
                      ? 'border-accent/45 bg-accent/12 text-accent'
                      : 'border-transparent text-muted hover:border-border hover:bg-panel-strong hover:text-foreground',
                  )}
                >
                  <span className="flex items-center gap-2.5">
                    <Icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </span>
                  <span className={cn('h-1.5 w-1.5 rounded-full bg-transparent', active && 'pulse-dot bg-accent')} />
                </Link>
              );
            })}
          </nav>

          <div className="mt-4 rounded-2xl border border-border bg-panel-strong p-3">
            <div className="flex items-center justify-between">
              <p className="text-[11px] uppercase tracking-[0.15em] text-muted">Pipeline</p>
              <Badge variant="positive">Live</Badge>
            </div>
            <div className="mt-3 space-y-1.5 text-xs text-muted">
              <p>Objective: return</p>
              <p>Refresh: daily 09:00 HKT</p>
            </div>
          </div>
        </aside>

        <main className="space-y-4">
          <header className="reveal-up rounded-3xl border border-border/70 bg-panel p-4 shadow-[0_12px_24px_rgba(4,6,14,0.3)]">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-[11px] uppercase tracking-[0.15em] text-muted">Workspace</p>
                <p className="mt-1 font-display text-3xl">{activeItem?.label ?? 'Command'}</p>
              </div>
              <div className="rounded-xl border border-border bg-panel-strong px-3 py-2 text-xs text-muted">
                Use `x-plan-id` to simulate tiers locally.
              </div>
            </div>
          </header>
          {children}
        </main>
      </div>
    </div>
  );
}
