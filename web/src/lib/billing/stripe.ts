import Stripe from 'stripe';

import type { PlanId } from '@/lib/types';

export type BillingInterval = 'monthly' | 'annual';

type PaidPlanId = Exclude<PlanId, 'free'>;

function readEnv(name: string): string | null {
  const value = process.env[name];
  if (!value || value.trim().length === 0) {
    return null;
  }
  return value.trim();
}

function isTruthy(value: string | null): boolean {
  if (!value) {
    return false;
  }
  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}

export function getStripeClient(): Stripe | null {
  const secretKey = readEnv('STRIPE_SECRET_KEY');
  if (!secretKey) {
    return null;
  }
  return new Stripe(secretKey);
}

export function getStripePriceId(plan: PaidPlanId, interval: BillingInterval): string | null {
  if (plan === 'pro') {
    return interval === 'monthly' ? readEnv('STRIPE_PRICE_PRO_MONTHLY') : readEnv('STRIPE_PRICE_PRO_ANNUAL');
  }
  return interval === 'monthly' ? readEnv('STRIPE_PRICE_ELITE_MONTHLY') : readEnv('STRIPE_PRICE_ELITE_ANNUAL');
}

export function resolveAppUrl(request: Request): string {
  const configured = readEnv('NEXT_PUBLIC_APP_URL') ?? readEnv('APP_URL');
  if (configured) {
    return configured;
  }

  if (isTruthy(readEnv('TRUST_REQUEST_HOST_HEADERS'))) {
    const origin = request.headers.get('origin');
    if (origin) {
      return origin;
    }

    const host = request.headers.get('x-forwarded-host') ?? request.headers.get('host');
    const proto = request.headers.get('x-forwarded-proto') ?? 'https';
    if (host) {
      return `${proto}://${host}`;
    }
  }

  return 'http://localhost:3000';
}
