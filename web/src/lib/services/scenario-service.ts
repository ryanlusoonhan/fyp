import { optimizeDecisionThreshold } from '@/lib/domain/signal-engine';
import { getSignalHistory } from '@/lib/data/signal-repository';
import type { Objective, ScenarioRunInput, ScenarioRunResult } from '@/lib/types';

export async function runScenario(input: ScenarioRunInput): Promise<ScenarioRunResult> {
  const history = await getSignalHistory(120);

  const probsUp = history.map((signal) => signal.probBuy);
  const yTrue = history.map((signal, idx) => {
    const transformed = signal.probBuy + Math.cos(idx * 0.4) * 0.09;
    return transformed > 0.5 ? 1 : 0;
  });
  const alignedFutureReturns = history.map((signal, idx) => {
    return Math.sin(idx * 0.6) * 0.04 + (signal.probBuy - 0.5) * 0.08;
  });

  const thresholds = buildThresholdGrid(input.thresholdMin, input.thresholdMax, input.step);

  const result = optimizeDecisionThreshold({
    objective: input.objective,
    probsUp,
    yTrue,
    alignedFutureReturns,
    thresholds,
    barrierWindow: input.barrierWindow,
    cost: input.cost,
  });

  return {
    objective: input.objective,
    bestThreshold: result.bestThreshold,
    bestScore: result.bestScore,
    candidates: result.candidates,
  };
}

export function buildThresholdGrid(minValue: number, maxValue: number, step: number): number[] {
  const min = Math.max(0.05, minValue);
  const max = Math.min(0.95, maxValue);
  const stride = step <= 0 ? 0.01 : step;

  const values: number[] = [];
  for (let value = min; value <= max + 1e-9; value += stride) {
    values.push(Number(value.toFixed(4)));
  }

  return values.length > 0 ? values : [0.5];
}

export function normalizeObjective(objectiveRaw: string | null | undefined): Objective {
  return objectiveRaw === 'f1' ? 'f1' : 'return';
}
