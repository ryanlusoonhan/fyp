import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getSignalHistory } from '@/lib/data/signal-repository';
import { toPercent } from '@/lib/utils';

export default async function SignalsPage() {
  const history = await getSignalHistory(40);

  return (
    <Card>
      <CardTitle>Signal history</CardTitle>
      <CardDescription className="mt-1">Recent 40 weekly outputs from the active model.</CardDescription>

      <div className="mt-5 overflow-x-auto border border-border">
        <table className="w-full text-left text-sm">
          <thead className="bg-panel-strong font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
            <tr>
              <th className="px-4 py-3">Date</th>
              <th className="px-4 py-3">Signal</th>
              <th className="px-4 py-3">Prob(BUY)</th>
              <th className="px-4 py-3">Threshold</th>
              <th className="px-4 py-3">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {history.map((signal) => (
              <tr key={signal.id} className="border-t border-border/70 bg-panel">
                <td className="px-4 py-3 font-mono text-xs text-slate-300">{signal.asOfDate}</td>
                <td className="px-4 py-3">
                  <Badge variant={signal.classification === 'BUY' ? 'positive' : 'negative'}>
                    {signal.classification}
                  </Badge>
                </td>
                <td className="px-4 py-3 font-semibold">{toPercent(signal.probBuy, 2)}</td>
                <td className="px-4 py-3 font-mono text-xs text-slate-300">{signal.threshold.toFixed(2)}</td>
                <td className="px-4 py-3 capitalize text-slate-200">{signal.confidenceBand}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
