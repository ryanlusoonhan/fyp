import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

import { Card, CardDescription, CardTitle } from '@/components/ui/card';

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-shell text-foreground">
      <main className="mx-auto max-w-4xl space-y-5 px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-wrap items-center justify-between gap-3 border border-border bg-panel p-3">
          <Link href="/" className="inline-flex items-center gap-2 text-sm text-muted transition hover:text-foreground">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Link>
          <Link href="/dashboard" className="border border-border bg-panel-strong px-3 py-1.5 text-sm text-muted hover:border-accent/40 hover:text-foreground">
            Open interface
          </Link>
        </header>

        <Card className="border-accent/45 bg-panel-strong">
          <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Internal deployment</p>
          <CardTitle className="mt-2 text-4xl">Pricing Disabled</CardTitle>
          <CardDescription className="mt-2">
            This repository is now configured as a private HSI analytics tool for internal use only.
            Subscription plans and billing workflows are intentionally disabled.
          </CardDescription>
          <div className="mt-4 space-y-2 text-sm text-slate-200">
            <p>- All analytics pages are available without tier gating.</p>
            <p>- Live refresh, explainability, scenario lab, and history are available directly.</p>
            <p>- Use the dashboard navigation to access all modules.</p>
          </div>
        </Card>
      </main>
    </div>
  );
}
