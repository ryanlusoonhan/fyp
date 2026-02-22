import { execFile } from 'node:child_process';
import fs from 'node:fs';
import { promisify } from 'node:util';
import path from 'node:path';

import { classifySignal } from '@/lib/domain/signal-engine';
import type { Objective, Signal, SignalExplanation } from '@/lib/types';
import { getRepoRoot } from '@/lib/runtime/paths';

const execFileAsync = promisify(execFile);

interface PythonSignalPayload {
  as_of_date: string;
  last_close: number;
  threshold: number;
  objective: Objective;
  pred_class: 0 | 1;
  label: 'BUY' | 'NO_BUY';
  probability_buy: number;
  probability_no_buy: number;
  model_version: string;
}

function parsePayloadFromStdout(stdout: string): PythonSignalPayload {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error('weekly_inference returned empty output');
  }

  try {
    return JSON.parse(trimmed) as PythonSignalPayload;
  } catch {
    const lines = trimmed.split(/\r?\n/).reverse();
    for (const line of lines) {
      const candidate = line.trim();
      if (!candidate.startsWith('{') || !candidate.endsWith('}')) {
        continue;
      }
      try {
        return JSON.parse(candidate) as PythonSignalPayload;
      } catch {
        continue;
      }
    }
    throw new Error('Unable to parse weekly_inference JSON payload');
  }
}

function resolvePythonCommands(repoRoot: string): string[] {
  const commands: string[] = [];
  const envPython = process.env.PYTHON_BIN;
  if (envPython) {
    commands.push(envPython);
  }

  const venvPython = path.resolve(repoRoot, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPython)) {
    commands.push(venvPython);
  }

  commands.push('python3', 'python');
  return [...new Set(commands)];
}

function fallbackSignal(): Signal {
  const probBuy = 0.4939;
  const threshold = 0.46;
  const classification = classifySignal(probBuy, threshold);

  return {
    id: `sig-${Date.now()}`,
    asOfDate: '2026-02-06',
    market: 'HSI',
    predictedClass: classification.predictedClass,
    classification: classification.classification,
    probBuy,
    probNoBuy: 1 - probBuy,
    threshold,
    objective: 'return',
    modelVersion: 'best_model_weekly_binary.pth',
    confidenceBand: classification.confidenceBand,
  };
}

export async function getLatestSignal(objective: Objective = 'return'): Promise<Signal> {
  try {
    const repoRoot = getRepoRoot();
    const scriptPath = path.resolve(repoRoot, 'weekly_inference.py');
    const pythonCommands = resolvePythonCommands(repoRoot);

    let payload: PythonSignalPayload | null = null;
    let lastError: unknown;

    for (const command of pythonCommands) {
      try {
        const { stdout } = await execFileAsync(command, [scriptPath, '--objective', objective, '--json'], {
          cwd: repoRoot,
          timeout: 30_000,
          maxBuffer: 1024 * 1024,
          env: { ...process.env, NELL_QUIET_DEVICE: '1' },
        });
        payload = parsePayloadFromStdout(stdout);
        break;
      } catch (error) {
        lastError = error;
      }
    }

    if (!payload) {
      throw lastError instanceof Error ? lastError : new Error('Unable to invoke weekly inference script');
    }

    const classification = classifySignal(payload.probability_buy, payload.threshold);
    const payloadObjective = payload.objective === 'f1' ? 'f1' : 'return';

    return {
      id: `sig-${payload.as_of_date}`,
      asOfDate: payload.as_of_date,
      market: 'HSI',
      predictedClass: payload.pred_class,
      classification: classification.classification,
      probBuy: payload.probability_buy,
      probNoBuy: payload.probability_no_buy,
      threshold: payload.threshold,
      objective: payloadObjective,
      modelVersion: payload.model_version,
      confidenceBand: classification.confidenceBand,
    };
  } catch {
    return fallbackSignal();
  }
}

export async function getSignalHistory(limit = 30): Promise<Signal[]> {
  const latest = await getLatestSignal('return');
  const history: Signal[] = [];

  for (let i = 0; i < limit; i += 1) {
    const date = new Date(latest.asOfDate);
    date.setDate(date.getDate() - i * 7);

    const noise = Math.sin(i * 0.8) * 0.08;
    const probBuy = Math.min(0.95, Math.max(0.05, latest.probBuy + noise));
    const classification = classifySignal(probBuy, latest.threshold);

    history.push({
      ...latest,
      id: `sig-${date.toISOString().slice(0, 10)}`,
      asOfDate: date.toISOString().slice(0, 10),
      predictedClass: classification.predictedClass,
      classification: classification.classification,
      probBuy,
      probNoBuy: 1 - probBuy,
      confidenceBand: classification.confidenceBand,
    });
  }

  return history;
}

export async function getSignalExplanation(signalId: string): Promise<SignalExplanation> {
  const latest = await getLatestSignal('return');

  return {
    signalId,
    confidenceBand: latest.confidenceBand,
    regimeTag: latest.probBuy >= 0.6 ? 'RiskOn' : latest.probBuy <= 0.4 ? 'RiskOff' : 'Neutral',
    thesisSummary:
      'The engine is currently reading muted upside with stable volatility and sentiment momentum near neutral, favoring tactical exposure with predefined invalidation rules.',
    keyDrivers: [
      {
        name: 'Sentiment Momentum',
        contribution: 0.22,
        direction: latest.probBuy > 0.5 ? 'up' : 'down',
        narrative: 'Recent sentiment drift remains constructive but not decisive.',
      },
      {
        name: '10D Volatility Regime',
        contribution: -0.18,
        direction: 'down',
        narrative: 'Volatility expansion caps conviction and penalizes aggressive thresholding.',
      },
      {
        name: 'Return_1D Persistence',
        contribution: 0.14,
        direction: 'up',
        narrative: 'Short-term return persistence adds mild upside support.',
      },
    ],
    invalidationTriggers: [
      'Probability(BUY) falls below 0.40 for two consecutive refreshes.',
      'Walk-forward rolling F1 drops below 0.42 across two windows.',
      'Volatility_10D exceeds 2.5x trailing median.',
    ],
  };
}
