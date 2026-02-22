import Link from 'next/link';
import { ArrowLeft, CheckCircle2 } from 'lucide-react';

import { PricingGridClient } from '@/components/marketing/pricing-grid-client';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';

const POINTS = [
  'No lock-in',
  'Cancel anytime',
  'Educational analytics only',
];

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-shell text-foreground">
      <div className="glass-grid absolute inset-0 -z-10" />
      <main className="mx-auto max-w-6xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/" className="inline-flex items-center gap-2 text-sm text-muted transition hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Link>
          <Link
            href="/dashboard"
            className="rounded-xl border border-border bg-panel-strong px-4 py-2 text-sm text-muted transition hover:border-accent/35 hover:text-foreground"
          >
            Open App
          </Link>
        </header>

        <section className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
          <Card className="reveal-up border-accent/40 bg-[linear-gradient(145deg,rgba(245,165,36,0.18),rgba(16,21,34,0.92)_45%,rgba(111,168,255,0.1))]">
            <p className="text-[11px] uppercase tracking-[0.15em] text-amber-200">Pricing</p>
            <CardTitle className="mt-3 text-5xl leading-[1.03] sm:text-6xl">Pick Your Depth</CardTitle>
            <CardDescription className="mt-3 text-slate-200">
              Free for visibility. Pro and Elite for conviction.
            </CardDescription>
          </Card>

          <Card className="reveal-up">
            <CardTitle className="text-xl">Simple Terms</CardTitle>
            <ul className="mt-4 space-y-3 text-sm text-slate-200">
              {POINTS.map((point) => (
                <li key={point} className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-amber-200" />
                  {point}
                </li>
              ))}
            </ul>
          </Card>
        </section>

        <PricingGridClient />
      </main>
    </div>
  );
}
