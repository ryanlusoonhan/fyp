import { NextResponse } from 'next/server';
import { z } from 'zod';

import { getRequestContext } from '@/lib/api/auth';
import { getTier } from '@/lib/domain/pricing';
import { getStripeClient, getStripePriceId, resolveAppUrl, type BillingInterval } from '@/lib/billing/stripe';

export const runtime = 'nodejs';

const payloadSchema = z.object({
  plan: z.enum(['pro', 'elite']),
  interval: z.enum(['monthly', 'annual']).default('monthly'),
});

function intervalLabel(interval: BillingInterval): string {
  return interval === 'annual' ? 'annual' : 'monthly';
}

export async function POST(request: Request) {
  const payload = await request.json().catch(() => ({}));
  const parsed = payloadSchema.safeParse(payload);

  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid checkout payload.' }, { status: 400 });
  }

  const { plan, interval } = parsed.data;
  const tier = getTier(plan);

  const stripe = getStripeClient();
  const priceId = getStripePriceId(plan, interval);
  const appUrl = resolveAppUrl(request);
  const auth = await getRequestContext();

  if (!stripe || !priceId) {
    return NextResponse.json(
      {
        data: {
          checkoutMode: 'stub',
          tier,
          interval,
          setupRequired: true,
          message:
            'Stripe is not fully configured. Add STRIPE_SECRET_KEY and plan price IDs to enable live checkout.',
        },
      },
      { status: 200 },
    );
  }

  const successUrl = new URL('/pricing?checkout=success', appUrl).toString();
  const cancelUrl = new URL('/pricing?checkout=canceled', appUrl).toString();

  let session;
  try {
    session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: successUrl,
      cancel_url: cancelUrl,
      metadata: {
        plan,
        interval: intervalLabel(interval),
        source: 'nell-terminal',
        userId: auth.userId ?? 'anonymous',
      },
      allow_promotion_codes: true,
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'Stripe session creation failed.';
    return NextResponse.json({ error: reason }, { status: 500 });
  }

  return NextResponse.json({
    data: {
      checkoutMode: 'stripe',
      tier,
      interval,
      sessionId: session.id,
      url: session.url,
    },
  });
}
