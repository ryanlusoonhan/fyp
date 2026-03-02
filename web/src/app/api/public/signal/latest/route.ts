import { NextResponse } from 'next/server';

import { getLatestSignal } from '@/lib/data/signal-repository';

export async function GET() {
  const signal = await getLatestSignal('accuracy');

  return NextResponse.json({
    data: {
      asOfDate: signal.asOfDate,
      market: signal.market,
      confidenceBand: signal.confidenceBand,
      delayed: false,
      displayProbabilityBuy: Number((signal.probBuy * 100).toFixed(1)),
      displayThreshold: signal.threshold.toFixed(2),
      classification: signal.classification,
      objective: signal.objective,
    },
    disclaimer:
      'Internal analytics endpoint for reference use; outputs are decision-support only.',
  });
}
