import { describe, expect, it } from 'vitest';

import { parseWalkForwardCsv } from '@/lib/ingest/walk-forward-parser';
import type { WalkForwardWindow } from '@/lib/types';

describe('walk-forward parser', () => {
  it('parses windows and coerces numbers', () => {
    const csv = [
      'window_id,start_idx,end_idx,n_samples,accuracy,f1,ai_return_pct,buy_hold_return_pct',
      '1,0,100,100,0.45,0.57,18.78,16.06',
      '2,50,150,100,0.46,0.55,9.57,7.05',
    ].join('\n');

    const windows = parseWalkForwardCsv(csv);
    expect(windows).toHaveLength(2);

    const first = windows[0] as WalkForwardWindow;
    expect(first.windowId).toBe(1);
    expect(first.accuracy).toBeCloseTo(0.45, 8);
    expect(first.aiReturnPct).toBeCloseTo(18.78, 8);
  });

  it('returns empty array for empty content', () => {
    expect(parseWalkForwardCsv('')).toEqual([]);
  });
});
