import { NextResponse } from 'next/server';

import { getPerformanceSummary } from '@/lib/data/walk-forward-repository';

export async function GET() {
  const summary = await getPerformanceSummary();
  return NextResponse.json({ data: summary });
}
