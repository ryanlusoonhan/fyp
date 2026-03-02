import { NextResponse } from 'next/server';

import { getRequestContext } from '@/lib/api/auth';
import { getSignalExplanation } from '@/lib/data/signal-repository';

interface Params {
  params: Promise<{ signalId: string }>;
}

export async function GET(_: Request, { params }: Params) {
  const auth = await getRequestContext();

  const { signalId } = await params;
  const explanation = await getSignalExplanation(signalId);

  return NextResponse.json({ data: explanation, meta: { userId: auth.userId, authSource: auth.source } });
}
