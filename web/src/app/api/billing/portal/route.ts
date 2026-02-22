import { NextResponse } from 'next/server';
import { z } from 'zod';

import { getStripeClient, resolveAppUrl } from '@/lib/billing/stripe';

export const runtime = 'nodejs';

const payloadSchema = z.object({
  customerId: z.string().min(1),
});

export async function POST(request: Request) {
  const payload = await request.json().catch(() => ({}));
  const parsed = payloadSchema.safeParse(payload);

  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid portal payload.' }, { status: 400 });
  }

  const stripe = getStripeClient();
  if (!stripe) {
    return NextResponse.json(
      {
        data: {
          setupRequired: true,
          message: 'Stripe billing portal is unavailable until STRIPE_SECRET_KEY is configured.',
        },
      },
      { status: 200 },
    );
  }

  const appUrl = resolveAppUrl(request);
  const returnUrl = new URL('/pricing?portal=return', appUrl).toString();

  try {
    const session = await stripe.billingPortal.sessions.create({
      customer: parsed.data.customerId,
      return_url: returnUrl,
    });

    return NextResponse.json({
      data: {
        url: session.url,
      },
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'Unable to create billing portal session.';
    return NextResponse.json({ error: reason }, { status: 500 });
  }
}
