import type { PlanId, PricingTier } from '@/lib/types';

export const PRICING_TIERS: PricingTier[] = [
  {
    id: 'free',
    name: 'Free',
    priceMonthlyUsd: 0,
    description: 'Preview before committing.',
    cta: 'Start for Free',
    features: [
      '24h delayed signal',
      'Last 7 signals',
      'Public methodology',
      'Community updates',
    ],
  },
  {
    id: 'pro',
    name: 'Pro',
    priceMonthlyUsd: 39,
    description: 'Core weekly operating tier.',
    cta: 'Upgrade to Pro',
    highlighted: true,
    features: [
      'Live BUY / NO_BUY signal',
      'Full signal history',
      'Explainability cards',
      'Email and Telegram alerts',
      'Priority refresh queue',
    ],
  },
  {
    id: 'elite',
    name: 'Elite',
    priceMonthlyUsd: 79,
    description: 'Deep validation tier.',
    cta: 'Go Elite',
    features: [
      'Everything in Pro',
      'Scenario Lab',
      'Walk-forward deep dive',
      'CSV / PDF exports',
      'Priority support',
    ],
  },
];

export function getTier(plan: PlanId): PricingTier {
  const found = PRICING_TIERS.find((tier) => tier.id === plan);
  if (!found) {
    throw new Error(`Unknown plan: ${plan}`);
  }
  return found;
}
