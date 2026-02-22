import type { HTMLAttributes } from 'react';

import { cn } from '@/lib/utils';

type BadgeVariant = 'neutral' | 'positive' | 'warning' | 'negative';

const variantClasses: Record<BadgeVariant, string> = {
  neutral: 'border-border bg-panel-strong text-muted',
  positive: 'border-emerald-300/40 bg-emerald-400/12 text-emerald-200',
  warning: 'border-amber-300/40 bg-amber-400/14 text-amber-100',
  negative: 'border-rose-300/40 bg-rose-400/12 text-rose-200',
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ className, variant = 'neutral', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-sm border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.12em]',
        variantClasses[variant],
        className,
      )}
      {...props}
    />
  );
}
