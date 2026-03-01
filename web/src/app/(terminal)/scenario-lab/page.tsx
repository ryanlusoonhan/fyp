import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import { ScenarioLabClient } from '@/components/terminal/scenario-lab-client';
import { runScenario } from '@/lib/services/scenario-service';
import type { ScenarioRunResult } from '@/lib/types';

export default async function ScenarioLabPage() {
  let result: ScenarioRunResult;
  try {
    result = await runScenario({
      objective: 'return',
      thresholdMin: 0.3,
      thresholdMax: 0.7,
      step: 0.01,
      cost: 0.001,
      barrierWindow: 10,
    });
  } catch {
    result = {
      objective: 'return',
      bestThreshold: 0.5,
      bestScore: 0,
      candidates: [],
    };
  }

  return (
    <Card>
      <CardTitle>Scenario lab</CardTitle>
      <CardDescription className="mt-1">Threshold sensitivity and objective stress test.</CardDescription>
      <div className="mt-4">
        <ScenarioLabClient initialResult={result} />
      </div>
    </Card>
  );
}
