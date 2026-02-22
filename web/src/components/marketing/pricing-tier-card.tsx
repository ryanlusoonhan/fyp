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
          ? 'border-accent/60 bg-panel-strong'
          : 'bg-panel'
      }`}
    >
      <div>
        <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">{tier.name}</p>
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
            <CheckCircle2 className="mt-0.5 h-4 w-4 text-accent" />
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
