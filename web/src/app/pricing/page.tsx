import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

import { PricingGridClient } from '@/components/marketing/pricing-grid-client';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-shell text-foreground">
      <main className="mx-auto max-w-6xl space-y-5 px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-wrap items-center justify-between gap-3 border border-border bg-panel p-3">
          <Link href="/" className="inline-flex items-center gap-2 text-sm text-muted transition hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Link>
          <Link href="/dashboard" className="border border-border bg-panel-strong px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-foreground">
            Open interface
          </Link>
        </header>

        <section className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
          <Card className="reveal-up border-accent/45 bg-panel-strong">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Pricing</p>
            <CardTitle className="mt-2 text-4xl">Stock Prediction Interface Plans</CardTitle>
            <CardDescription className="mt-2">
              Start on Free. Upgrade when you need real-time access and deeper validation tools.
            </CardDescription>
          </Card>

          <Card className="reveal-up">
            <CardTitle>Policy</CardTitle>
            <div className="mt-3 space-y-2 font-mono text-[11px] text-muted">
              <p>- No long-term contract</p>
              <p>- Cancel anytime</p>
              <p>- Educational analytics only</p>
            </div>
          </Card>
        </section>

        <PricingGridClient />
      </main>
    </div>
  );
}
