import { describe, expect, it } from 'vitest';

import { parsePayloadFromStdout, parseSignalHistoryCsv } from '@/lib/data/signal-repository';

describe('signal-repository', () => {
  it('parses inference payload with freshness fields', () => {
    const payload = parsePayloadFromStdout(
      JSON.stringify({
        as_of_date: '2026-03-01',
        last_close: 25000,
        threshold: 0.46,
        objective: 'return',
        pred_class: 1,
        label: 'BUY',
        probability_buy: 0.6,
        probability_no_buy: 0.4,
        model_version: 'best_model_weekly_binary.pth',
        data_status: 'fresh',
        last_refresh_at: '2026-03-01T12:00:00Z',
        latest_market_date: '2026-02-28',
      }),
    );

    expect(payload.data_status).toBe('fresh');
    expect(payload.latest_market_date).toBe('2026-02-28');
  });

  it('parses signal history csv rows', () => {
    const csv = [
      'as_of_date,last_close,threshold,objective,pred_class,label,probability_buy,probability_no_buy,model_version,data_status,last_refresh_at,latest_market_date',
      '2026-03-01,25000,0.46,return,1,BUY,0.6,0.4,best_model_weekly_binary.pth,fresh,2026-03-01T12:00:00Z,2026-02-28',
    ].join('\n');

    const rows = parseSignalHistoryCsv(csv);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.probability_buy).toBeCloseTo(0.6);
    expect(rows[0]?.data_status).toBe('fresh');
  });
});
