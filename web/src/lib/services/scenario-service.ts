import { execFile } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { promisify } from 'node:util';

import type { Objective, ScenarioRunInput, ScenarioRunResult } from '@/lib/types';
import { getRepoRoot } from '@/lib/runtime/paths';

const execFileAsync = promisify(execFile);

interface PythonScenarioResult {
  objective: Objective;
  bestThreshold: number;
  bestScore: number;
  candidates: Array<{
    threshold: number;
    score: number;
    buy_rate?: number;
    buyRate?: number;
  }>;
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

function parseScenarioPayload(stdout: string): PythonScenarioResult {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error('scenario_weekly.py returned empty output');
  }

  try {
    return JSON.parse(trimmed) as PythonScenarioResult;
  } catch {
    const lines = trimmed.split(/\r?\n/).reverse();
    for (const line of lines) {
      const candidate = line.trim();
      if (!candidate.startsWith('{') || !candidate.endsWith('}')) {
        continue;
      }
      try {
        return JSON.parse(candidate) as PythonScenarioResult;
      } catch {
        continue;
      }
    }
    throw new Error('Unable to parse scenario_weekly JSON payload');
  }
}

function mapScenarioResult(payload: PythonScenarioResult): ScenarioRunResult {
  return {
    objective: payload.objective === 'f1' ? 'f1' : 'return',
    bestThreshold: Number(payload.bestThreshold),
    bestScore: Number(payload.bestScore),
    candidates: (payload.candidates ?? []).map((candidate) => ({
      threshold: Number(candidate.threshold),
      score: Number(candidate.score),
      buyRate: Number(candidate.buyRate ?? candidate.buy_rate ?? 0),
    })),
  };
}

async function runPythonScenario(input: ScenarioRunInput): Promise<ScenarioRunResult> {
  const repoRoot = getRepoRoot();
  const scriptPath = path.resolve(repoRoot, 'scenario_weekly.py');
  const pythonCommands = resolvePythonCommands(repoRoot);
  let lastError: unknown;

  const args = [
    scriptPath,
    '--json',
    '--objective',
    input.objective,
    '--threshold-min',
    String(input.thresholdMin),
    '--threshold-max',
    String(input.thresholdMax),
    '--step',
    String(input.step),
    '--cost',
    String(input.cost),
    '--barrier-window',
    String(input.barrierWindow),
  ];

  for (const command of pythonCommands) {
    try {
      const { stdout } = await execFileAsync(command, args, {
        cwd: repoRoot,
        timeout: 120_000,
        maxBuffer: 1024 * 1024 * 2,
        env: { ...process.env, NELL_QUIET_DEVICE: '1' },
      });
      return mapScenarioResult(parseScenarioPayload(stdout));
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error('Unable to execute scenario_weekly.py');
}

export async function runScenario(input: ScenarioRunInput): Promise<ScenarioRunResult> {
  return runPythonScenario(input);
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
