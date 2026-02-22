import { WalkForwardChart } from '@/components/charts/walk-forward-chart';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { getWalkForwardWindows } from '@/lib/data/walk-forward-repository';

export default async function WalkForwardPage() {
  const windows = await getWalkForwardWindows();

  return (
    <Card>
      <CardTitle>Walk-Forward</CardTitle>
      <CardDescription className="mt-1">Window-by-window robustness check.</CardDescription>

      <div className="mt-5">
        <WalkForwardChart data={windows} />
      </div>
    </Card>
  );
}
