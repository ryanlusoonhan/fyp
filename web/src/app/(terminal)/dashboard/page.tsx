import { Activity, GaugeCircle, Layers3, TrendingUp } from 'lucide-react';

import { EquityCurveChart } from '@/components/charts/equity-curve-chart';
import { KpiCard } from '@/components/terminal/kpi-card';
import { SignalHeroCard } from '@/components/terminal/signal-hero-card';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { getLatestSignal } from '@/lib/data/signal-repository';
import { getPerformanceSummary, getWalkForwardWindows } from '@/lib/data/walk-forward-repository';
import { toPercent, toSignedPercent } from '@/lib/utils';

function buildEquitySeries(windows: Awaited<ReturnType<typeof getWalkForwardWindows>>) {
  let ai = 10_000;
  let benchmark = 10_000;

  return windows.map((window) => {
    ai *= 1 + window.aiReturnPct / 100;
    benchmark *= 1 + window.buyHoldReturnPct / 100;
    return {
      label: `W${window.windowId}`,
      ai,
      benchmark,
    };
  });
}

export default async function DashboardPage() {
  const [signal, summary, windows] = await Promise.all([
    getLatestSignal('return'),
    getPerformanceSummary(),
    getWalkForwardWindows(),
  ]);

  const equitySeries = buildEquitySeries(windows);

  return (
    <>
      <SignalHeroCard signal={signal} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Accuracy"
          value={toPercent(summary.latestAccuracy)}
          hint="Last 100 periods"
          icon={<Activity className="h-4 w-4" />}
        />
        <KpiCard
          label="F1"
          value={summary.latestF1.toFixed(3)}
          hint="BUY class quality"
          icon={<GaugeCircle className="h-4 w-4" />}
        />
        <KpiCard
          label="AI Return"
          value={toSignedPercent(summary.latestAiReturnPct)}
          hint={`vs B&H ${toSignedPercent(summary.latestBuyHoldReturnPct)}`}
          icon={<TrendingUp className="h-4 w-4" />}
        />
        <KpiCard
          label="Windows"
          value={String(windows.length)}
          hint="Walk-forward slices"
          icon={<Layers3 className="h-4 w-4" />}
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.7fr_1fr]">
        <Card>
          <CardTitle>Equity curve</CardTitle>
          <CardDescription className="mt-1">Compounded strategy return vs buy-and-hold.</CardDescription>
          <div className="mt-4">
            <EquityCurveChart data={equitySeries} />
          </div>
        </Card>

        <Card className="space-y-4">
          <CardTitle>Model monitor</CardTitle>
          <CardDescription>Operational checks for the current cycle.</CardDescription>

          <div className="space-y-3 text-sm text-slate-200">
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Optimization objective</p>
              <p className="mt-1 font-semibold">Return maximization</p>
            </div>
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Cadence</p>
              <p className="mt-1">Weekly decision, daily data refresh.</p>
            </div>
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Validation status</p>
              <p className="mt-1">Walk-forward metrics available and current.</p>
            </div>
          </div>
        </Card>
      </section>
    </>
  );
}
