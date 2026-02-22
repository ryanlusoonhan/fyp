import { NextRequest, NextResponse } from 'next/server';

import { canAccessFeature } from '@/lib/domain/entitlements';
import { getSignalHistory } from '@/lib/data/signal-repository';
import { getRequestContext } from '@/lib/api/auth';

export async function GET(request: NextRequest) {
  const auth = await getRequestContext();

  if (!canAccessFeature(auth.plan, 'signal-history-full')) {
    return NextResponse.json({ error: 'Upgrade required for full history.' }, { status: 403 });
  }

  const limitRaw = request.nextUrl.searchParams.get('limit');
  const limit = Number(limitRaw ?? '52');
  const boundedLimit = Number.isFinite(limit) ? Math.min(260, Math.max(1, limit)) : 52;

  const data = await getSignalHistory(boundedLimit);

  return NextResponse.json({
    data,
    meta: {
      plan: auth.plan,
      authSource: auth.source,
      count: data.length,
    },
  });
}
