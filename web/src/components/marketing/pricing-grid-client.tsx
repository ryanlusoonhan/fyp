'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';

import { CheckoutButton } from '@/components/marketing/checkout-button';
import { PricingTierCard } from '@/components/marketing/pricing-tier-card';
import { PRICING_TIERS } from '@/lib/domain/pricing';

type BillingInterval = 'monthly' | 'annual';

const intervals: Array<{ id: BillingInterval; label: string; hint: string }> = [
  { id: 'monthly', label: 'Monthly', hint: 'Pay as you go' },
  { id: 'annual', label: 'Annual', hint: 'Save ~17%' },
];

export function PricingGridClient() {
  const [interval, setInterval] = useState<BillingInterval>('monthly');

  const annualMultiplier = 10;
  const pricingRows = useMemo(() => {
    return PRICING_TIERS.map((tier) => {
      if (interval === 'monthly') {
        return {
          ...tier,
          displayPrice: tier.priceMonthlyUsd,
          cadenceLabel: '/mo',
        };
      }

      return {
        ...tier,
        displayPrice: tier.priceMonthlyUsd * annualMultiplier,
        cadenceLabel: '/yr',
      };
    });
  }, [interval]);

  return (
    <div className="space-y-6">
      <div className="mx-auto grid w-full max-w-xl grid-cols-2 rounded-2xl border border-border bg-panel p-1">
        {intervals.map((option) => {
          const active = option.id === interval;
          return (
            <button
              key={option.id}
              type="button"
              onClick={() => setInterval(option.id)}
              className={`rounded-xl px-4 py-3 text-left transition ${
                active ? 'bg-accent text-accent-foreground' : 'text-muted hover:bg-panel-strong hover:text-foreground'
              }`}
            >
              <p className="text-sm font-semibold">{option.label}</p>
              <p className={`text-xs ${active ? 'text-accent-foreground/80' : 'text-muted'}`}>{option.hint}</p>
            </button>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {pricingRows.map((tier) => {
          const priceNode = (
            <div>
              <p className="mt-2 font-display text-4xl">
                ${tier.displayPrice}
                <span className="ml-1 text-base text-muted">{tier.cadenceLabel}</span>
              </p>
              {interval === 'annual' && tier.priceMonthlyUsd > 0 ? (
                <p className="mt-1 text-xs text-emerald-300">Effective ${tier.priceMonthlyUsd}/mo paid yearly</p>
              ) : null}
            </div>
          );

          const cta =
            tier.id === 'free' ? (
              <Link
                href="/dashboard"
                className="inline-flex h-11 w-full items-center justify-center rounded-lg border border-border bg-panel text-sm font-semibold text-foreground transition hover:border-accent/40 hover:bg-panel-strong"
              >
                Start Free
              </Link>
            ) : (
              <CheckoutButton
                plan={tier.id}
                interval={interval}
                label={interval === 'annual' ? `${tier.cta} (Annual)` : tier.cta}
              />
            );

          return <PricingTierCard key={tier.id} tier={tier} cta={cta} priceOverride={priceNode} />;
        })}
      </div>
    </div>
  );
}
