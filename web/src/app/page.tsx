import Link from 'next/link';
import { ArrowRight, ChartCandlestick, Radar, ShieldCheck, TrendingUp } from 'lucide-react';

import { PricingTierCard } from '@/components/marketing/pricing-tier-card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { getLatestSignal } from '@/lib/data/signal-repository';
import { getPerformanceSummary } from '@/lib/data/walk-forward-repository';
import { PRICING_TIERS } from '@/lib/domain/pricing';
import { toPercent, toSignedPercent } from '@/lib/utils';

const PILLARS = [
  {
    title: 'Signal',
    description: 'Weekly BUY / NO_BUY with calibrated threshold.',
    icon: ChartCandlestick,
  },
  {
    title: 'Risk Context',
    description: 'Confidence + invalidation triggers in one view.',
    icon: ShieldCheck,
  },
  {
    title: 'Proof',
    description: 'Walk-forward windows and threshold simulation.',
    icon: Radar,
  },
];

export default async function HomePage() {
  const [signal, performance] = await Promise.all([
    getLatestSignal('return').catch(() => null),
    getPerformanceSummary().catch(() => null),
  ]);

  return (
    <div className="min-h-screen bg-shell text-foreground">
      <div className="glass-grid absolute inset-0 -z-10" />
      <main className="mx-auto max-w-6xl space-y-8 px-4 py-6 sm:px-6 lg:px-8">
        <header className="reveal-up rounded-3xl border border-border/70 bg-panel p-4 shadow-[0_16px_38px_rgba(3,4,10,0.45)]">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Link href="/" className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-content-center rounded-xl bg-[linear-gradient(140deg,#f5a524,#ffbe55)] text-accent-foreground">
                <TrendingUp className="h-5 w-5" />
              </div>
              <div>
                <p className="font-display text-lg">Nell Signal Terminal</p>
                <p className="text-[11px] uppercase tracking-[0.16em] text-muted">Weekly HSI Intelligence</p>
              </div>
            </Link>
            <div className="flex items-center gap-2">
              <Button asChild variant="ghost" size="sm">
                <Link href="/pricing">Pricing</Link>
              </Button>
              <Button asChild size="sm">
                <Link href="/dashboard">Open App</Link>
              </Button>
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[1.45fr_1fr]">
          <Card className="reveal-up border-accent/40 bg-[linear-gradient(145deg,rgba(245,165,36,0.22),rgba(15,19,30,0.94)_46%,rgba(111,168,255,0.12))]">
            <Badge variant="warning">Live Weekly Model</Badge>
            <CardTitle className="mt-4 text-5xl leading-[1.02] sm:text-6xl">Trade the Week. Not the Noise.</CardTitle>
            <CardDescription className="mt-4 max-w-2xl text-base text-slate-200">
              One clean weekly stance, backed by confidence context and walk-forward evidence.
            </CardDescription>
            <div className="mt-6 flex flex-wrap gap-3">
              <Button asChild size="lg">
                <Link href="/dashboard" className="inline-flex items-center gap-2">
                  Launch Terminal
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="secondary">
                <Link href="/pricing">See Plans</Link>
              </Button>
            </div>
          </Card>

          <Card className="reveal-up space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-[11px] uppercase tracking-[0.15em] text-muted">Live Snapshot</p>
              <Badge variant={signal?.classification === 'BUY' ? 'positive' : 'negative'}>
                {signal?.confidenceBand ?? 'low'}
              </Badge>
            </div>

            <div className="rounded-2xl border border-border bg-panel-strong p-4">
              <p className="text-[11px] uppercase tracking-[0.15em] text-muted">Current Stance</p>
              <p className="mt-1 font-display text-5xl">{signal?.classification ?? 'NO_BUY'}</p>
              <p className="mt-1 text-sm text-slate-200">
                Prob BUY {signal ? toPercent(signal.probBuy, 2) : '49.39%'}
              </p>
            </div>

            <div className="space-y-1 text-xs text-muted">
              <p>Date: {signal?.asOfDate ?? '2026-02-06'}</p>
              <p>Threshold: {signal?.threshold?.toFixed(2) ?? '0.46'}</p>
              <p>Model: {signal?.modelVersion ?? 'best_model_weekly_binary.pth'}</p>
            </div>
          </Card>
        </section>

        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="reveal-up">
            <p className="text-[11px] uppercase tracking-[0.15em] text-muted">Accuracy</p>
            <p className="mt-2 font-display text-4xl">{performance ? toPercent(performance.latestAccuracy) : '56.00%'}</p>
          </Card>
          <Card className="reveal-up">
            <p className="text-[11px] uppercase tracking-[0.15em] text-muted">F1</p>
            <p className="mt-2 font-display text-4xl">{performance?.latestF1?.toFixed(3) ?? '0.353'}</p>
          </Card>
          <Card className="reveal-up">
            <p className="text-[11px] uppercase tracking-[0.15em] text-muted">AI Return</p>
            <p className="mt-2 font-display text-4xl">
              {performance ? toSignedPercent(performance.latestAiReturnPct) : '+6.70%'}
            </p>
          </Card>
          <Card className="reveal-up">
            <p className="text-[11px] uppercase tracking-[0.15em] text-muted">B&H Return</p>
            <p className="mt-2 font-display text-4xl">
              {performance ? toSignedPercent(performance.latestBuyHoldReturnPct) : '+6.73%'}
            </p>
          </Card>
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          {PILLARS.map((pillar) => {
            const Icon = pillar.icon;
            return (
              <Card key={pillar.title} className="reveal-up space-y-3">
                <div className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-accent/35 bg-accent/10 text-amber-200">
                  <Icon className="h-5 w-5" />
                </div>
                <CardTitle>{pillar.title}</CardTitle>
                <CardDescription>{pillar.description}</CardDescription>
              </Card>
            );
          })}
        </section>

        <section className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.15em] text-muted">Plans</p>
              <h2 className="font-display text-3xl">Start Free. Upgrade for Depth.</h2>
            </div>
            <Link href="/pricing" className="text-sm text-amber-200 transition hover:text-amber-100">
              Full pricing
            </Link>
          </div>
          <div className="grid gap-4 lg:grid-cols-3">
            {PRICING_TIERS.map((tier) => (
              <PricingTierCard
                key={tier.id}
                tier={tier}
                cta={
                  <Button asChild className="w-full" variant={tier.highlighted ? 'primary' : 'secondary'}>
                    <Link href={tier.id === 'free' ? '/dashboard' : '/pricing'}>{tier.cta}</Link>
                  </Button>
                }
              />
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
