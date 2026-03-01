import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

import { getRequestContext } from '@/lib/api/auth';
import { getStripeClient, resolveAppUrl } from '@/lib/billing/stripe';

export const runtime = 'nodejs';

function readEnv(name: string): string | null {
  const value = process.env[name];
  if (!value || value.trim().length === 0) {
    return null;
  }
  return value.trim();
}

async function resolveCustomerIdForUser(userId: string): Promise<string | null> {
  const supabaseUrl = readEnv('NEXT_PUBLIC_SUPABASE_URL');
  const serviceRole = readEnv('SUPABASE_SERVICE_ROLE_KEY');
  if (!supabaseUrl || !serviceRole) {
    return null;
  }

  const adminClient = createClient(supabaseUrl, serviceRole, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { data } = await adminClient
    .from('subscriptions')
    .select('stripe_customer_id,current_period_end')
    .eq('user_id', userId)
    .eq('status', 'active')
    .not('stripe_customer_id', 'is', null)
    .order('current_period_end', { ascending: false })
    .limit(1)
    .maybeSingle();

  return data?.stripe_customer_id ?? null;
}

export async function POST(request: Request) {
  const auth = await getRequestContext();
  if (!auth.userId) {
    return NextResponse.json({ error: 'Authentication required.' }, { status: 401 });
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
  const customerId = await resolveCustomerIdForUser(auth.userId);
  if (!customerId) {
    return NextResponse.json({ error: 'No active billing customer found.' }, { status: 404 });
  }

  try {
    const session = await stripe.billingPortal.sessions.create({
      customer: customerId,
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
