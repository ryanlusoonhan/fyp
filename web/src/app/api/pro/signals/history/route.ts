import { NextRequest, NextResponse } from 'next/server';

import { getSignalHistory } from '@/lib/data/signal-repository';
import { getRequestContext } from '@/lib/api/auth';

export async function GET(request: NextRequest) {
  const auth = await getRequestContext();

  const limitRaw = request.nextUrl.searchParams.get('limit');
  const limit = Number(limitRaw ?? '52');
  const boundedLimit = Number.isFinite(limit) ? Math.min(260, Math.max(1, limit)) : 52;

  const data = await getSignalHistory(boundedLimit);

  return NextResponse.json({
    data,
    meta: {
      userId: auth.userId,
      authSource: auth.source,
      count: data.length,
    },
  });
}
