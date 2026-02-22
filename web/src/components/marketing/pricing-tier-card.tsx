import { CheckCircle2 } from 'lucide-react';
import type { ReactNode } from 'react';

import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';
import type { PricingTier } from '@/lib/types';

interface PricingTierCardProps {
  tier: PricingTier;
  cta?: ReactNode;
  priceOverride?: ReactNode;
}

export function PricingTierCard({ tier, cta, priceOverride }: PricingTierCardProps) {
  return (
    <Card
      className={`flex h-full flex-col gap-4 ${
        tier.highlighted
          ? 'border-accent/60 bg-[linear-gradient(145deg,rgba(245,165,36,0.18),rgba(18,22,34,0.92)_48%,rgba(111,168,255,0.1))]'
          : 'bg-panel'
      }`}
    >
      <div>
        <p className="text-[11px] uppercase tracking-[0.15em] text-muted">{tier.name}</p>
        {priceOverride ?? (
          <CardTitle className="mt-2 text-4xl">
            ${tier.priceMonthlyUsd}
            <span className="ml-1 text-base text-muted">/mo</span>
          </CardTitle>
        )}
        <CardDescription className="mt-2">{tier.description}</CardDescription>
      </div>

      <ul className="space-y-2 text-sm text-slate-200">
        {tier.features.map((feature) => (
          <li key={feature} className="flex items-start gap-2">
            <CheckCircle2 className="mt-0.5 h-4 w-4 text-amber-200" />
            <span>{feature}</span>
          </li>
        ))}
      </ul>

      <div className="mt-auto">
        {cta ?? (
          <Button className="w-full" variant={tier.highlighted ? 'primary' : 'secondary'}>
            {tier.cta}
          </Button>
        )}
      </div>
    </Card>
  );
}
