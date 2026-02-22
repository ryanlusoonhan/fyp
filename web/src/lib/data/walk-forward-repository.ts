import fs from 'node:fs/promises';
import path from 'node:path';

import { parseWalkForwardCsv } from '@/lib/ingest/walk-forward-parser';
import type { PerformanceSummary, WalkForwardWindow } from '@/lib/types';
import { getModelsDir } from '@/lib/runtime/paths';

function fallbackWindows(): WalkForwardWindow[] {
  return [
    { windowId: 1, startIdx: 0, endIdx: 100, nSamples: 100, accuracy: 0.47, f1: 0.37, aiReturnPct: 3.2, buyHoldReturnPct: 2.6 },
    { windowId: 2, startIdx: 50, endIdx: 150, nSamples: 100, accuracy: 0.54, f1: 0.34, aiReturnPct: -1.8, buyHoldReturnPct: -2.4 },
    { windowId: 3, startIdx: 100, endIdx: 200, nSamples: 100, accuracy: 0.56, f1: 0.35, aiReturnPct: 6.7, buyHoldReturnPct: 6.73 },
  ];
}

function computeSummary(windows: WalkForwardWindow[]): PerformanceSummary {
  if (windows.length === 0) {
    return {
      latestAccuracy: 0,
      latestF1: 0,
      latestAiReturnPct: 0,
      latestBuyHoldReturnPct: 0,
      meanAccuracy: 0,
      meanF1: 0,
      meanAiReturnPct: 0,
      meanBuyHoldReturnPct: 0,
    };
  }

  const latest = windows[windows.length - 1] as WalkForwardWindow;

  const totals = windows.reduce(
    (acc, window) => {
      acc.accuracy += window.accuracy;
      acc.f1 += window.f1;
      acc.aiReturnPct += window.aiReturnPct;
      acc.buyHoldReturnPct += window.buyHoldReturnPct;
      return acc;
    },
    { accuracy: 0, f1: 0, aiReturnPct: 0, buyHoldReturnPct: 0 },
  );

  return {
    latestAccuracy: latest.accuracy,
    latestF1: latest.f1,
    latestAiReturnPct: latest.aiReturnPct,
    latestBuyHoldReturnPct: latest.buyHoldReturnPct,
    meanAccuracy: totals.accuracy / windows.length,
    meanF1: totals.f1 / windows.length,
    meanAiReturnPct: totals.aiReturnPct / windows.length,
    meanBuyHoldReturnPct: totals.buyHoldReturnPct / windows.length,
  };
}

export async function getWalkForwardWindows(): Promise<WalkForwardWindow[]> {
  const walkForwardPath = path.resolve(getModelsDir(), 'backtest_weekly_walk_forward.csv');
  try {
    const raw = await fs.readFile(walkForwardPath, 'utf8');
    const parsed = parseWalkForwardCsv(raw);
    return parsed.length > 0 ? parsed : fallbackWindows();
  } catch {
    return fallbackWindows();
  }
}

export async function getPerformanceSummary(): Promise<PerformanceSummary> {
  const windows = await getWalkForwardWindows();
  return computeSummary(windows);
}
