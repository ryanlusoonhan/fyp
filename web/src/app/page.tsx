import Link from 'next/link';
import { ArrowRight, BarChart3, FileSearch, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { getLatestSignal } from '@/lib/data/signal-repository';
import { getPerformanceSummary } from '@/lib/data/walk-forward-repository';
import { toPercent, toSignedPercent } from '@/lib/utils';

const MODULES = [
  {
    title: 'Signal Engine',
    detail: 'Weekly BUY/NO_BUY with threshold and confidence.',
    icon: BarChart3,
  },
  {
    title: 'Explainability',
    detail: 'Driver-level context and invalidation triggers.',
    icon: FileSearch,
  },
  {
    title: 'Validation Layer',
    detail: 'Walk-forward and scenario stress testing.',
    icon: ShieldCheck,
  },
];

export default async function HomePage() {
  const [signal, performance] = await Promise.all([
    getLatestSignal('return').catch(() => null),
    getPerformanceSummary().catch(() => null),
  ]);

  return (
    <div className="min-h-screen bg-shell text-foreground">
      <main className="mx-auto max-w-6xl space-y-5 px-4 py-5 sm:px-6 lg:px-8">
        <header className="reveal-up border border-border bg-panel p-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="font-display text-2xl">Stock Prediction Interface</p>
              <p className="mt-2 max-w-2xl text-sm text-muted">
                Professional dashboard for weekly signal output, confidence context, and validation analytics.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button asChild size="sm" variant="secondary">
                <Link href="/pricing">Pricing</Link>
              </Button>
              <Button asChild size="sm">
                <Link href="/dashboard">Open interface</Link>
              </Button>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[1.35fr_1fr]">
          <Card className="reveal-up border-accent/50 bg-panel-strong">
            <div className="flex items-center justify-between gap-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Current weekly signal</p>
              <Badge variant={signal?.classification === 'BUY' ? 'positive' : 'negative'}>
                {signal?.confidenceBand ?? 'low'}
              </Badge>
            </div>
            <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
              <div>
                <p className="font-display text-5xl">{signal?.classification ?? 'NO_BUY'}</p>
                <p className="mt-1 font-mono text-xs text-muted">
                  as_of={signal?.asOfDate ?? '2026-02-06'} threshold={signal?.threshold?.toFixed(2) ?? '0.46'}
                </p>
              </div>
              <div className="text-right">
                <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">probability buy</p>
                <p className="font-display text-4xl">{signal ? toPercent(signal.probBuy, 2) : '49.39%'}</p>
              </div>
            </div>
          </Card>

          <Card className="reveal-up">
            <CardTitle>Runtime status</CardTitle>
            <CardDescription className="mt-2">Model and pipeline context for this session.</CardDescription>
            <div className="mt-4 space-y-2 font-mono text-xs text-muted">
              <p>market=HSI</p>
              <p>objective=return</p>
              <p>cadence=weekly signal / daily refresh</p>
              <p>model={signal?.modelVersion ?? 'best_model_weekly_binary.pth'}</p>
            </div>
          </Card>
        </section>

        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="reveal-up">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Accuracy</p>
            <p className="mt-2 font-display text-4xl">{performance ? toPercent(performance.latestAccuracy) : '56.00%'}</p>
          </Card>
          <Card className="reveal-up">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">F1 score</p>
            <p className="mt-2 font-display text-4xl">{performance?.latestF1?.toFixed(3) ?? '0.353'}</p>
          </Card>
          <Card className="reveal-up">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">AI return</p>
            <p className="mt-2 font-display text-4xl">
              {performance ? toSignedPercent(performance.latestAiReturnPct) : '+6.70%'}
            </p>
          </Card>
          <Card className="reveal-up">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">B&H return</p>
            <p className="mt-2 font-display text-4xl">
              {performance ? toSignedPercent(performance.latestBuyHoldReturnPct) : '+6.73%'}
            </p>
          </Card>
        </section>

        <section className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
          <Card className="reveal-up">
            <CardTitle>Core modules</CardTitle>
            <div className="mt-4 space-y-3">
              {MODULES.map((module) => {
                const Icon = module.icon;
                return (
                  <div key={module.title} className="flex items-start gap-3 border border-border bg-panel-strong p-3">
                    <Icon className="mt-0.5 h-4 w-4 text-accent" />
                    <div>
                      <p className="text-sm font-semibold">{module.title}</p>
                      <p className="text-xs text-muted">{module.detail}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          <Card className="reveal-up">
            <CardTitle>Primary actions</CardTitle>
            <div className="mt-4 space-y-2">
              <Link href="/dashboard" className="flex items-center justify-between border border-border bg-panel-strong p-3 text-sm hover:border-accent/40">
                Open dashboard
                <ArrowRight className="h-4 w-4 text-muted" />
              </Link>
              <Link href="/signals" className="flex items-center justify-between border border-border bg-panel-strong p-3 text-sm hover:border-accent/40">
                View signal history
                <ArrowRight className="h-4 w-4 text-muted" />
              </Link>
              <Link href="/walk-forward" className="flex items-center justify-between border border-border bg-panel-strong p-3 text-sm hover:border-accent/40">
                Check walk-forward
                <ArrowRight className="h-4 w-4 text-muted" />
              </Link>
            </div>
          </Card>
        </section>
      </main>
    </div>
  );
}
