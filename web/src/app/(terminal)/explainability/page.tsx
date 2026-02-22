import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getLatestSignal, getSignalExplanation } from '@/lib/data/signal-repository';

export default async function ExplainabilityPage() {
  const latest = await getLatestSignal('return');
  const explanation = await getSignalExplanation(latest.id);

  return (
    <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
      <Card>
        <CardTitle>Driver Stack</CardTitle>
        <CardDescription className="mt-1">What pushed this signal.</CardDescription>

        <div className="mt-5 space-y-3">
          {explanation.keyDrivers.map((driver) => (
            <div key={driver.name} className="rounded-xl border border-border bg-panel-strong p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold text-slate-100">{driver.name}</p>
                <Badge variant={driver.direction === 'up' ? 'positive' : 'warning'}>
                  {driver.direction === 'up' ? '+' : '-'}{Math.round(driver.contribution * 100)} bps
                </Badge>
              </div>
              <p className="mt-2 text-sm text-slate-300">{driver.narrative}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card className="space-y-4">
        <CardTitle>Regime + Invalidation</CardTitle>
        <CardDescription>{explanation.thesisSummary}</CardDescription>

        <div className="rounded-xl border border-border bg-panel-strong p-3 text-sm">
          <p className="text-[11px] uppercase tracking-[0.14em] text-muted">Regime</p>
          <p className="mt-1 font-semibold">{explanation.regimeTag}</p>
        </div>

        <div>
          <p className="text-[11px] uppercase tracking-[0.14em] text-muted">Invalidation</p>
          <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-200">
            {explanation.invalidationTriggers.map((trigger) => (
              <li key={trigger}>{trigger}</li>
            ))}
          </ul>
        </div>
      </Card>
    </div>
  );
}
