import { cva, type VariantProps } from 'class-variance-authority';
import { cloneElement, isValidElement } from 'react';
import type { ButtonHTMLAttributes, ReactElement } from 'react';

import { cn } from '@/lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap rounded-md border text-sm font-semibold transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/70 focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
  {
    variants: {
      variant: {
        primary: 'border-accent/80 bg-accent text-accent-foreground hover:brightness-110',
        secondary: 'border-border bg-panel-strong text-foreground hover:border-accent/40',
        ghost: 'border-transparent bg-transparent text-muted hover:bg-panel-strong hover:text-foreground',
      },
      size: {
        sm: 'h-9 px-3 text-xs',
        md: 'h-11 px-4',
        lg: 'h-12 px-6',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  },
);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  };

export function Button({ className, variant, size, asChild = false, children, ...props }: ButtonProps) {
  const sharedClassName = cn(buttonVariants({ variant, size }), className);

  if (asChild && isValidElement(children)) {
    const child = children as ReactElement<{ className?: string }>;
    return cloneElement(child, {
      className: cn(sharedClassName, child.props.className),
    });
  }

  return (
    <button className={sharedClassName} {...props}>
      {children}
    </button>
  );
}
