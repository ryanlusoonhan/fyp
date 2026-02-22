import { NextResponse } from 'next/server';

import { getLatestSignal } from '@/lib/data/signal-repository';

export async function GET() {
  const signal = await getLatestSignal('return');

  return NextResponse.json({
    data: {
      asOfDate: signal.asOfDate,
      market: signal.market,
      confidenceBand: signal.confidenceBand,
      delayed: true,
      displayProbabilityBuy: Number((signal.probBuy * 100).toFixed(1)),
      displayThreshold: 'Locked',
      lockedFields: ['classification', 'predictedClass', 'threshold', 'objective'],
    },
    disclaimer:
      'Free preview data is delayed and partially masked. Upgrade to Pro for live explainable signals.',
  });
}
