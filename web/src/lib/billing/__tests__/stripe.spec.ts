import { afterEach, describe, expect, it } from 'vitest';

import { getStripePriceId, resolveAppUrl } from '@/lib/billing/stripe';

const originalEnv = { ...process.env };

afterEach(() => {
  process.env = { ...originalEnv };
});

describe('stripe billing utilities', () => {
  it('resolves plan-specific price ids', () => {
    process.env.STRIPE_PRICE_PRO_MONTHLY = 'price_pro_monthly';
    process.env.STRIPE_PRICE_PRO_ANNUAL = 'price_pro_annual';
    process.env.STRIPE_PRICE_ELITE_MONTHLY = 'price_elite_monthly';
    process.env.STRIPE_PRICE_ELITE_ANNUAL = 'price_elite_annual';

    expect(getStripePriceId('pro', 'monthly')).toBe('price_pro_monthly');
    expect(getStripePriceId('pro', 'annual')).toBe('price_pro_annual');
    expect(getStripePriceId('elite', 'monthly')).toBe('price_elite_monthly');
    expect(getStripePriceId('elite', 'annual')).toBe('price_elite_annual');
  });

  it('resolves app url with environment precedence', () => {
    process.env.NEXT_PUBLIC_APP_URL = 'https://env.example.com';
    const request = new Request('http://localhost:3000/pricing');
    expect(resolveAppUrl(request)).toBe('https://env.example.com');
  });

  it('falls back to request origin headers', () => {
    delete process.env.NEXT_PUBLIC_APP_URL;
    delete process.env.APP_URL;

    const requestWithOrigin = new Request('http://localhost:3000/pricing', {
      headers: { origin: 'https://origin.example.com' },
    });
    expect(resolveAppUrl(requestWithOrigin)).toBe('https://origin.example.com');

    const requestWithForwarded = new Request('http://localhost:3000/pricing', {
      headers: { 'x-forwarded-proto': 'https', 'x-forwarded-host': 'nell.example.com' },
    });
    expect(resolveAppUrl(requestWithForwarded)).toBe('https://nell.example.com');
  });
});
