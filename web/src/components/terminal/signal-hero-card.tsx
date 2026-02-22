import { ArrowDownRight, ArrowUpRight } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import type { Signal } from '@/lib/types';
import { toPercent } from '@/lib/utils';

export function SignalHeroCard({ signal }: { signal: Signal }) {
  const isBuy = signal.classification === 'BUY';

  return (
    <Card className="reveal-up border-accent/45 bg-panel-strong">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Latest signal</p>
          <CardTitle className="mt-2 flex items-center gap-2 text-4xl">
            {signal.classification}
            {isBuy ? <ArrowUpRight className="h-8 w-8 text-emerald-300" /> : <ArrowDownRight className="h-8 w-8 text-rose-300" />}
          </CardTitle>
          <CardDescription className="mt-1 text-xs text-slate-200">
            {signal.asOfDate} · {signal.modelVersion}
          </CardDescription>
        </div>

        <div className="min-w-[200px] border border-border bg-panel p-3 text-right">
          <Badge variant={isBuy ? 'positive' : 'negative'}>{signal.confidenceBand}</Badge>
          <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Probability buy</p>
          <p className="font-display text-3xl">{toPercent(signal.probBuy, 2)}</p>
          <p className="font-mono text-[11px] text-muted">Threshold {signal.threshold.toFixed(2)}</p>
        </div>
      </div>
    </Card>
  );
}
