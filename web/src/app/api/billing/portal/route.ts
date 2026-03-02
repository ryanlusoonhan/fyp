import { NextResponse } from 'next/server';

export const runtime = 'nodejs';

export async function POST() {
  return NextResponse.json(
    {
      error: 'Billing portal is disabled for this internal analytics deployment.',
    },
    { status: 410 },
  );
}
