import { WalkForwardChart } from '@/components/charts/walk-forward-chart';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { getWalkForwardWindows } from '@/lib/data/walk-forward-repository';

export default async function WalkForwardPage() {
  const windows = await getWalkForwardWindows();

  return (
    <Card>
      <CardTitle>Walk-forward validation</CardTitle>
      <CardDescription className="mt-1">Window-level return comparison for robustness monitoring.</CardDescription>

      <div className="mt-5">
        <WalkForwardChart data={windows} />
      </div>
    </Card>
  );
}
