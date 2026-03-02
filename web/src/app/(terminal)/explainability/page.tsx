import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getLatestSignal, getSignalExplanation } from '@/lib/data/signal-repository';

export default async function ExplainabilityPage() {
  const latest = await getLatestSignal('accuracy');
  const explanation = await getSignalExplanation(latest.id);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <Card>
          <CardTitle>Main reasons behind the signal</CardTitle>
          <CardDescription className="mt-1">Higher contribution means the factor had more influence.</CardDescription>

          <div className="mt-5 space-y-3">
            {explanation.keyDrivers.map((driver) => (
              <div key={driver.name} className="border border-border bg-panel-strong p-3">
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
          <CardTitle>Signal safety checks</CardTitle>
          <CardDescription>{explanation.thesisSummary}</CardDescription>

          <div className="border border-border bg-panel-strong p-3 text-sm">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Regime</p>
            <p className="mt-1 font-semibold">{explanation.regimeTag}</p>
          </div>

          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Invalidation triggers</p>
            <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-slate-200">
              {explanation.invalidationTriggers.map((trigger) => (
                <li key={trigger}>{trigger}</li>
              ))}
            </ul>
          </div>
        </Card>
      </div>

      {explanation.marketContext ? (
        <Card>
          <CardTitle>OpenBB context (explainability only)</CardTitle>
          <CardDescription className="mt-1">
            OpenBB is not part of model training right now. It is used for market context and data reliability.
          </CardDescription>
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Training mode</p>
              <p className="mt-1 text-sm">{explanation.marketContext.trainingMode}</p>
            </div>
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">OpenBB provider</p>
              <p className="mt-1 text-sm">{explanation.marketContext.openbbProvider ?? 'unknown'}</p>
            </div>
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Refresh status</p>
              <p className="mt-1 text-sm capitalize">{explanation.marketContext.openbbStatus ?? 'unknown'}</p>
            </div>
            <div className="border border-border bg-panel-strong p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">Latest market date</p>
              <p className="mt-1 text-sm">{explanation.marketContext.latestMarketDate ?? 'unknown'}</p>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted">{explanation.marketContext.note}</p>
        </Card>
      ) : null}
    </div>
  );
}
