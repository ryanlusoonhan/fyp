export type Objective = 'f1' | 'return';

export type ConfidenceBand = 'low' | 'medium' | 'high';

export type SignalClassification = 'BUY' | 'NO_BUY';

export interface ClassificationResult {
  predictedClass: 0 | 1;
  classification: SignalClassification;
  confidenceBand: ConfidenceBand;
  confidenceDistance: number;
}

export interface ThresholdCandidate {
  threshold: number;
  score: number;
  buyRate: number;
}

export interface OptimizeThresholdInput {
  objective: Objective;
  probsUp: number[];
  yTrue: number[];
  thresholds?: number[];
  alignedFutureReturns?: number[];
  barrierWindow?: number;
  cost?: number;
}

export interface OptimizeThresholdResult {
  bestThreshold: number;
  bestScore: number;
  candidates: ThresholdCandidate[];
}

export function classifySignal(probBuy: number, threshold: number): ClassificationResult {
  const predictedClass = probBuy > threshold ? 1 : 0;
  const distance = Math.abs(probBuy - threshold);

  let confidenceBand: ConfidenceBand = 'low';
  if (distance >= 0.2) {
    confidenceBand = 'high';
  } else if (predictedClass === 1 && distance >= 0.05) {
    confidenceBand = 'medium';
  } else if (predictedClass === 0 && distance >= 0.15) {
    confidenceBand = 'medium';
  }

  return {
    predictedClass,
    classification: predictedClass === 1 ? 'BUY' : 'NO_BUY',
    confidenceBand,
    confidenceDistance: distance,
  };
}

function f1Score(yTrue: number[], yPred: number[]): number {
  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < yTrue.length; i += 1) {
    const y = yTrue[i] ?? 0;
    const p = yPred[i] ?? 0;
    if (p === 1 && y === 1) tp += 1;
    if (p === 1 && y === 0) fp += 1;
    if (p === 0 && y === 1) fn += 1;
  }

  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  if (precision + recall === 0) return 0;
  return (2 * precision * recall) / (precision + recall);
}

function simulateStrategyReturn(
  predClass: number[],
  alignedFutureReturns: number[],
  barrierWindow: number,
  cost: number,
): number {
  let balance = 1;
  const step = Math.max(1, barrierWindow);

  for (let t = 0; t < predClass.length; t += step) {
    const action = predClass[t] ?? 0;
    const r = alignedFutureReturns[t] ?? 0;

    if (action === 1) {
      balance *= 1 - cost;
      balance *= 1 + r;
      balance *= 1 - cost;
    }
  }

  return balance - 1;
}

export function optimizeDecisionThreshold(input: OptimizeThresholdInput): OptimizeThresholdResult {
  const {
    objective,
    probsUp,
    yTrue,
    alignedFutureReturns,
    barrierWindow = 10,
    cost = 0.001,
  } = input;

  if (objective === 'return' && !alignedFutureReturns) {
    throw new Error('alignedFutureReturns is required for return objective');
  }

  const thresholds =
    input.thresholds && input.thresholds.length > 0
      ? input.thresholds
      : Array.from({ length: 41 }, (_, idx) => Number((0.3 + idx * 0.01).toFixed(2)));

  let bestThreshold = thresholds[0] ?? 0.5;
  let bestScore = Number.NEGATIVE_INFINITY;

  const candidates: ThresholdCandidate[] = thresholds.map((threshold) => {
    const predClass = probsUp.map((p) => (p > threshold ? 1 : 0));
    const buyRate = predClass.length === 0 ? 0 : predClass.filter((x) => x === 1).length / predClass.length;

    const score =
      objective === 'f1'
        ? f1Score(yTrue, predClass)
        : simulateStrategyReturn(predClass, alignedFutureReturns ?? [], barrierWindow, cost);

    if (score > bestScore) {
      bestScore = score;
      bestThreshold = threshold;
    }

    return {
      threshold,
      score,
      buyRate,
    };
  });

  return {
    bestThreshold,
    bestScore,
    candidates,
  };
}
