import { WalkForwardChart } from '@/components/charts/walk-forward-chart';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { getWalkForwardWindows } from '@/lib/data/walk-forward-repository';
import { toPercent, toSignedPercent } from '@/lib/utils';

export default async function WalkForwardPage() {
  const windows = await getWalkForwardWindows();

  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>Walk-forward validation</CardTitle>
        <CardDescription className="mt-1">Window-level return, accuracy, and F1 comparison for robustness monitoring.</CardDescription>

        <div className="mt-5">
          <WalkForwardChart data={windows} />
        </div>

        <div className="mt-5 overflow-x-auto border border-border">
          <table className="w-full text-left text-sm">
            <thead className="bg-panel-strong font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
              <tr>
                <th className="px-4 py-3">Window</th>
                <th className="px-4 py-3">Accuracy</th>
                <th className="px-4 py-3">F1</th>
                <th className="px-4 py-3">AI Return</th>
                <th className="px-4 py-3">B&H Return</th>
              </tr>
            </thead>
            <tbody>
              {windows.map((window) => (
                <tr key={window.windowId} className="border-t border-border/70 bg-panel">
                  <td className="px-4 py-3 font-mono text-xs text-slate-300">W{window.windowId}</td>
                  <td className="px-4 py-3">{toPercent(window.accuracy, 2)}</td>
                  <td className="px-4 py-3">{window.f1.toFixed(3)}</td>
                  <td className="px-4 py-3">{toSignedPercent(window.aiReturnPct, 2)}</td>
                  <td className="px-4 py-3">{toSignedPercent(window.buyHoldReturnPct, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
