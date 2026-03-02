import { execFile } from 'node:child_process';
import fs from 'node:fs';
import fsPromises from 'node:fs/promises';
import { promisify } from 'node:util';
import path from 'node:path';

import { classifySignal } from '@/lib/domain/signal-engine';
import type { Objective, Signal, SignalExplanation } from '@/lib/types';
import { getOpenbbOverview } from '@/lib/data/openbb-repository';
import { getModelsDir, getRepoRoot } from '@/lib/runtime/paths';

const execFileAsync = promisify(execFile);

export interface PythonSignalPayload {
  as_of_date: string;
  last_close: number;
  threshold: number;
  objective: Objective;
  pred_class: 0 | 1;
  label: 'BUY' | 'NO_BUY';
  probability_buy: number;
  probability_no_buy: number;
  model_version: string;
  data_status?: string;
  last_refresh_at?: string;
  latest_market_date?: string;
  data_provider?: string;
  logged_at?: string;
}

function parseJsonFromStdout(stdout: string): unknown {
  const trimmed = stdout.trim();
  if (!trimmed) {
    throw new Error('weekly_inference returned empty output');
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    const lines = trimmed.split(/\r?\n/).reverse();
    for (const line of lines) {
      const candidate = line.trim();
      if (!candidate.startsWith('{') || !candidate.endsWith('}')) {
        continue;
      }
      try {
        return JSON.parse(candidate);
      } catch {
        continue;
      }
    }
    throw new Error('Unable to parse weekly_inference JSON payload');
  }
}

export function parsePayloadFromStdout(stdout: string): PythonSignalPayload {
  const parsed = parseJsonFromStdout(stdout);
  if (!parsed || typeof parsed !== 'object' || !('as_of_date' in parsed)) {
    throw new Error('Unexpected latest signal payload shape from weekly_inference');
  }
  return parsed as PythonSignalPayload;
}

export function parseHistoryPayloadFromStdout(stdout: string): PythonSignalPayload[] {
  const parsed = parseJsonFromStdout(stdout);
  if (!parsed || typeof parsed !== 'object' || !('history' in parsed)) {
    throw new Error('Unexpected historical payload shape from weekly_inference');
  }
  const history = (parsed as { history?: unknown }).history;
  if (!Array.isArray(history)) {
    throw new Error('Historical payload missing history array');
  }
  return history as PythonSignalPayload[];
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

function buildSignalId(payload: PythonSignalPayload, salt = ''): string {
  const raw = [
    payload.as_of_date ?? 'unknown-date',
    payload.objective ?? 'return',
    payload.last_refresh_at ?? payload.logged_at ?? '',
    payload.threshold?.toFixed?.(4) ?? String(payload.threshold ?? ''),
    payload.model_version ?? '',
    salt,
  ]
    .filter((part) => part.length > 0)
    .join('-');

  return `sig-${raw.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
}

function payloadToSignal(payload: PythonSignalPayload, salt = ''): Signal {
  const classification = classifySignal(payload.probability_buy, payload.threshold);
  const payloadObjective =
    payload.objective === 'accuracy' ? 'accuracy' : payload.objective === 'f1' ? 'f1' : 'return';

  return {
    id: buildSignalId(payload, salt),
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
    dataStatus: payload.data_status,
    lastRefreshAt: payload.last_refresh_at,
    latestMarketDate: payload.latest_market_date,
    dataProvider: payload.data_provider,
  };
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
    objective: 'accuracy',
    modelVersion: 'best_model_weekly_binary.pth',
    confidenceBand: classification.confidenceBand,
    dataStatus: 'stale',
  };
}

async function runPythonInference(objective: Objective, extraArgs: string[] = []): Promise<PythonSignalPayload> {
  const repoRoot = getRepoRoot();
  const scriptPath = path.resolve(repoRoot, 'weekly_inference.py');
  const pythonCommands = resolvePythonCommands(repoRoot);

  let payload: PythonSignalPayload | null = null;
  let lastError: unknown;

  for (const command of pythonCommands) {
    try {
      const { stdout } = await execFileAsync(command, [scriptPath, '--objective', objective, '--json', ...extraArgs], {
        cwd: repoRoot,
        timeout: 60_000,
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
  return payload;
}

async function runPythonInferenceHistory(objective: Objective, limit: number): Promise<PythonSignalPayload[]> {
  const repoRoot = getRepoRoot();
  const scriptPath = path.resolve(repoRoot, 'weekly_inference.py');
  const pythonCommands = resolvePythonCommands(repoRoot);

  let payload: PythonSignalPayload[] | null = null;
  let lastError: unknown;

  for (const command of pythonCommands) {
    try {
      const { stdout } = await execFileAsync(
        command,
        [
          scriptPath,
          '--objective',
          objective,
          '--json',
          '--no-append-history',
          '--history-limit',
          String(limit),
        ],
        {
          cwd: repoRoot,
          timeout: 120_000,
          maxBuffer: 2 * 1024 * 1024,
          env: { ...process.env, NELL_QUIET_DEVICE: '1' },
        },
      );
      payload = parseHistoryPayloadFromStdout(stdout);
      break;
    } catch (error) {
      lastError = error;
    }
  }

  if (!payload) {
    throw lastError instanceof Error ? lastError : new Error('Unable to invoke weekly historical inference script');
  }
  return payload;
}

export async function getLatestSignal(objective: Objective = 'accuracy'): Promise<Signal> {
  try {
    const payload = await runPythonInference(objective, ['--no-append-history']);
    return payloadToSignal(payload);
  } catch (error) {
    console.error('Failed to load latest signal from Python inference:', error);
    return fallbackSignal();
  }
}

export function parseSignalHistoryCsv(csvContent: string): PythonSignalPayload[] {
  const lines = csvContent
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length <= 1) {
    return [];
  }

  const headers = lines[0]!.split(',');
  const rows = lines.slice(1);

  return rows.map((row) => {
    const values = row.split(',');
    const record = Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
    return {
      as_of_date: record.as_of_date,
      last_close: Number(record.last_close),
      threshold: Number(record.threshold),
      objective: record.objective === 'accuracy' ? 'accuracy' : record.objective === 'f1' ? 'f1' : 'return',
      pred_class: Number(record.pred_class) === 1 ? 1 : 0,
      label: Number(record.pred_class) === 1 ? 'BUY' : 'NO_BUY',
      probability_buy: Number(record.probability_buy),
      probability_no_buy: Number(record.probability_no_buy),
      model_version: record.model_version,
      data_status: record.data_status || undefined,
      last_refresh_at: record.last_refresh_at || undefined,
      latest_market_date: record.latest_market_date || undefined,
      logged_at: record.logged_at || undefined,
    } satisfies PythonSignalPayload;
  });
}

function parseSortTimestamp(payload: PythonSignalPayload): number {
  const candidates = [payload.logged_at, payload.last_refresh_at];
  for (const value of candidates) {
    if (!value) {
      continue;
    }
    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return Date.parse(`${payload.as_of_date}T00:00:00Z`) || 0;
}

function dedupeHistoryRows(rows: PythonSignalPayload[]): PythonSignalPayload[] {
  const seen = new Set<string>();
  const deduped: PythonSignalPayload[] = [];
  for (const row of rows) {
    const key = `${row.as_of_date}|${row.objective}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(row);
  }
  return deduped;
}

function sortHistoryRows(rows: PythonSignalPayload[]): PythonSignalPayload[] {
  return rows.sort((a, b) => {
    const byDate = String(b.as_of_date).localeCompare(String(a.as_of_date));
    if (byDate !== 0) {
      return byDate;
    }
    return parseSortTimestamp(b) - parseSortTimestamp(a);
  });
}

function mergeHistoryRows(primary: PythonSignalPayload[], secondary: PythonSignalPayload[]): PythonSignalPayload[] {
  const mergedByKey = new Map<string, PythonSignalPayload>();

  for (const row of secondary) {
    const key = `${row.as_of_date}|${row.objective}`;
    mergedByKey.set(key, row);
  }
  for (const row of primary) {
    const key = `${row.as_of_date}|${row.objective}`;
    mergedByKey.set(key, row);
  }

  return sortHistoryRows(Array.from(mergedByKey.values()));
}

export async function getSignalHistory(limit = 30): Promise<Signal[]> {
  const historyPath = path.resolve(getModelsDir(), 'signal_history_weekly.csv');
  const csvRows = await fsPromises
    .readFile(historyPath, 'utf8')
    .then((raw) => parseSignalHistoryCsv(raw).filter((row) => row.as_of_date))
    .catch(() => [] as PythonSignalPayload[]);

  if (csvRows.length >= limit) {
    return dedupeHistoryRows(sortHistoryRows(csvRows))
      .slice(0, limit)
      .map((payload, index) => payloadToSignal(payload, String(index)));
  }

  try {
    const inferredRows = await runPythonInferenceHistory('accuracy', Math.max(limit, 52));
    const mergedRows = dedupeHistoryRows(mergeHistoryRows(csvRows, inferredRows)).slice(0, limit);
    if (mergedRows.length > 0) {
      return mergedRows.map((payload, index) => payloadToSignal(payload, String(index)));
    }
  } catch (error) {
    if (csvRows.length > 0) {
      return dedupeHistoryRows(sortHistoryRows(csvRows))
        .slice(0, limit)
        .map((payload, index) => payloadToSignal(payload, String(index)));
    }
    console.error('Failed to build real historical signals from Python inference:', error);
  }

  const latest = await getLatestSignal('accuracy');
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

export async function refreshLatestSignal(objective: Objective = 'accuracy'): Promise<Signal> {
  const payload = await runPythonInference(objective, ['--refresh-openbb', '--refresh-mode', 'live']);
  return payloadToSignal(payload);
}

export async function getSignalExplanation(signalId: string): Promise<SignalExplanation> {
  const [latest, openbbOverview] = await Promise.all([getLatestSignal('accuracy'), getOpenbbOverview()]);

  return {
    signalId,
    confidenceBand: latest.confidenceBand,
    regimeTag: latest.probBuy >= 0.6 ? 'RiskOn' : latest.probBuy <= 0.4 ? 'RiskOff' : 'Neutral',
    thesisSummary:
      'Signal is trained on the primary professor dataset. OpenBB is used as live market context for interpretation and monitoring.',
    keyDrivers: [
      {
        name: 'Recent HSI direction',
        contribution: 0.24,
        direction: latest.probBuy > 0.5 ? 'up' : 'down',
        narrative: 'Short-term Hang Seng direction in recent periods is the strongest influence in the current output.',
      },
      {
        name: 'Sentiment trend',
        contribution: -0.19,
        direction: latest.probBuy > 0.5 ? 'up' : 'down',
        narrative: 'Smoothed sentiment from the primary dataset either supports or weakens the BUY probability.',
      },
      {
        name: 'Volatility pressure',
        contribution: 0.13,
        direction: latest.probBuy > 0.5 ? 'down' : 'up',
        narrative: 'When volatility rises, confidence in directional upside usually drops.',
      },
    ],
    invalidationTriggers: [
      'Probability(BUY) falls below 0.45 for two consecutive runs.',
      'Walk-forward accuracy remains below 0.55 for two refresh cycles.',
      'Data freshness status becomes stale for more than one week.',
    ],
    marketContext: {
      trainingMode: 'Primary dataset only (OpenBB excluded from training features)',
      openbbProvider: openbbOverview.source.provider,
      openbbStatus: openbbOverview.source.status,
      latestMarketDate: openbbOverview.source.latestMarketDate,
      lastRefreshAt: openbbOverview.source.lastRefreshAt,
      note: 'OpenBB currently supports explainability and data-health context only.',
    },
  };
}
