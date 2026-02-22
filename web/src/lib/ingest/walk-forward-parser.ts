import type { WalkForwardWindow } from '@/lib/types';

export function parseWalkForwardCsv(csvContent: string): WalkForwardWindow[] {
  if (!csvContent.trim()) {
    return [];
  }

  const lines = csvContent
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length <= 1) {
    return [];
  }

  const rows = lines.slice(1);

  return rows.map((row) => {
    const [windowId, startIdx, endIdx, nSamples, accuracy, f1, aiReturnPct, buyHoldReturnPct] = row.split(',');

    return {
      windowId: Number(windowId),
      startIdx: Number(startIdx),
      endIdx: Number(endIdx),
      nSamples: Number(nSamples),
      accuracy: Number(accuracy),
      f1: Number(f1),
      aiReturnPct: Number(aiReturnPct),
      buyHoldReturnPct: Number(buyHoldReturnPct),
    } satisfies WalkForwardWindow;
  });
}
