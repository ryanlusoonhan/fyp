import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { ScenarioLabClient } from '@/components/terminal/scenario-lab-client';
import { runScenario } from '@/lib/services/scenario-service';
import type { ScenarioRunResult } from '@/lib/types';

export default async function ScenarioLabPage() {
  let result: ScenarioRunResult;
  try {
    result = await runScenario({
      objective: 'accuracy',
      thresholdMin: 0.3,
      thresholdMax: 0.7,
      step: 0.01,
      cost: 0.001,
      barrierWindow: 10,
    });
  } catch {
    result = {
      objective: 'accuracy',
      bestThreshold: 0.5,
      bestScore: 0,
      candidates: [],
    };
  }

  return (
    <Card>
      <CardTitle>Threshold tuning</CardTitle>
      <CardDescription className="mt-1">
        Test different thresholds and objectives to see how decision quality changes.
      </CardDescription>
      <div className="mt-4">
        <ScenarioLabClient initialResult={result} />
      </div>
    </Card>
  );
}
