import { describe, expect, it } from 'vitest';

import {
  classifySignal,
  optimizeDecisionThreshold,
  type Objective,
} from '@/lib/domain/signal-engine';

describe('signal-engine', () => {
  it('classifies BUY when probability is above threshold', () => {
    const out = classifySignal(0.61, 0.55);
    expect(out.classification).toBe('BUY');
    expect(out.predictedClass).toBe(1);
    expect(out.confidenceBand).toBe('medium');
  });

  it('classifies NO_BUY when probability is below threshold', () => {
    const out = classifySignal(0.42, 0.55);
    expect(out.classification).toBe('NO_BUY');
    expect(out.predictedClass).toBe(0);
    expect(out.confidenceBand).toBe('low');
  });

  it('optimizes threshold for f1 objective', () => {
    const probsUp = [0.2, 0.4, 0.6, 0.8];
    const yTrue = [0, 0, 1, 1];
    const thresholds = [0.3, 0.5, 0.7];

    const result = optimizeDecisionThreshold({
      objective: 'f1',
      probsUp,
      yTrue,
      thresholds,
    });

    expect(result.bestThreshold).toBe(0.5);
    expect(result.bestScore).toBe(1);
  });

  it('optimizes threshold for return objective', () => {
    const probsUp = [0.49, 0.51, 0.52, 0.9];
    const yTrue = [0, 1, 1, 1];
    const alignedFutureReturns = [0.1, -0.2, 0.15, 0.02];

    const result = optimizeDecisionThreshold({
      objective: 'return',
      probsUp,
      yTrue,
      alignedFutureReturns,
      thresholds: [0.5, 0.55],
      barrierWindow: 1,
      cost: 0,
    });

    expect(result.bestThreshold).toBe(0.55);
    expect(result.bestScore).toBeGreaterThan(0);
  });

  it('throws when return objective has no returns array', () => {
    expect(() =>
      optimizeDecisionThreshold({
        objective: 'return' as Objective,
        probsUp: [0.1],
        yTrue: [0],
      }),
    ).toThrow(/alignedFutureReturns/i);
  });
});
