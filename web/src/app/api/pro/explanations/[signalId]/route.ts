import { NextResponse } from 'next/server';

import { getRequestContext } from '@/lib/api/auth';
import { canAccessFeature } from '@/lib/domain/entitlements';
import { getSignalExplanation } from '@/lib/data/signal-repository';

interface Params {
  params: Promise<{ signalId: string }>;
}

export async function GET(_: Request, { params }: Params) {
  const auth = await getRequestContext();

  if (!canAccessFeature(auth.plan, 'advanced-explainability')) {
    return NextResponse.json({ error: 'Upgrade required for explainability.' }, { status: 403 });
  }

  const { signalId } = await params;
  const explanation = await getSignalExplanation(signalId);

  return NextResponse.json({ data: explanation, meta: { plan: auth.plan, authSource: auth.source } });
}
