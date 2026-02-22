'use client';

import { useState } from 'react';
import { ArrowRight, LoaderCircle } from 'lucide-react';

import { Button } from '@/components/ui/button';

type PaidPlan = 'pro' | 'elite';
type BillingInterval = 'monthly' | 'annual';

interface CheckoutButtonProps {
  plan: PaidPlan;
  interval: BillingInterval;
  label: string;
}

export function CheckoutButton({ plan, interval, label }: CheckoutButtonProps) {
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function handleCheckout() {
    setLoading(true);
    setMessage(null);

    try {
      const response = await fetch('/api/billing/checkout', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({ plan, interval }),
      });

      const payload = (await response.json()) as {
        error?: string;
        data?: { url?: string; message?: string };
      };

      if (!response.ok) {
        setMessage(payload.error ?? 'Checkout failed.');
        return;
      }

      if (payload.data?.url) {
        window.location.href = payload.data.url;
        return;
      }

      setMessage(payload.data?.message ?? 'Checkout is not configured yet.');
    } catch {
      setMessage('Network error.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-2">
      <Button
        className="w-full"
        variant="primary"
        onClick={handleCheckout}
        disabled={loading}
        aria-busy={loading}
      >
        {loading ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <ArrowRight className="mr-2 h-4 w-4" />}
        {label}
      </Button>
      {message ? <p className="text-xs text-amber-200">{message}</p> : null}
    </div>
  );
}
