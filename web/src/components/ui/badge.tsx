import type { HTMLAttributes } from 'react';

import { cn } from '@/lib/utils';

type BadgeVariant = 'neutral' | 'positive' | 'warning' | 'negative';

const variantClasses: Record<BadgeVariant, string> = {
  neutral: 'border-border bg-panel-strong text-muted',
  positive: 'border-emerald-300/45 bg-emerald-400/14 text-emerald-200',
  warning: 'border-amber-300/45 bg-amber-400/16 text-amber-100',
  negative: 'border-rose-300/45 bg-rose-400/16 text-rose-200',
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ className, variant = 'neutral', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em]',
        variantClasses[variant],
        className,
      )}
      {...props}
    />
  );
}
