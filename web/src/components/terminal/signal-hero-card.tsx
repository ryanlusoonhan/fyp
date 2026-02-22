import { ArrowDownRight, ArrowUpRight } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import type { Signal } from '@/lib/types';
import { toPercent } from '@/lib/utils';

export function SignalHeroCard({ signal }: { signal: Signal }) {
  const isBuy = signal.classification === 'BUY';

  return (
    <Card className="reveal-up relative overflow-hidden border-accent/40 bg-[linear-gradient(135deg,rgba(245,165,36,0.2),rgba(18,22,34,0.94)_42%,rgba(111,168,255,0.14))]">
      <div className="absolute -right-14 -top-14 h-44 w-44 rounded-full bg-accent/20 blur-3xl" />
      <div className="relative flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.15em] text-amber-200">Latest Signal</p>
          <CardTitle className="mt-3 flex items-center gap-2 text-5xl">
            {signal.classification}
            {isBuy ? <ArrowUpRight className="h-8 w-8 text-emerald-300" /> : <ArrowDownRight className="h-8 w-8 text-rose-300" />}
          </CardTitle>
          <CardDescription className="mt-2 text-xs text-slate-200">
            {signal.asOfDate} · {signal.modelVersion}
          </CardDescription>
        </div>

        <div className="min-w-[180px] rounded-2xl border border-border/80 bg-panel/70 p-3 text-right">
          <Badge variant={isBuy ? 'positive' : 'negative'}>{signal.confidenceBand}</Badge>
          <p className="mt-3 text-[11px] uppercase tracking-[0.14em] text-muted">Prob. Buy</p>
          <p className="font-display text-4xl">{toPercent(signal.probBuy, 2)}</p>
          <p className="text-xs text-muted">Threshold {signal.threshold.toFixed(2)}</p>
        </div>
      </div>
    </Card>
  );
}
