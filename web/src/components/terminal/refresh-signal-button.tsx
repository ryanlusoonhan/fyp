'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { RefreshCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';

export function RefreshSignalButton() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRefresh = async () => {
    if (isRefreshing) {
      return;
    }
    setError(null);
    setIsRefreshing(true);
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      const devPlanId = process.env.NEXT_PUBLIC_DEV_PLAN_ID ?? 'elite';
      if (process.env.NODE_ENV !== 'production') {
        headers['x-plan-id'] = devPlanId;
      }

      const response = await fetch('/api/signal/refresh', {
        method: 'POST',
        headers,
        body: JSON.stringify({ objective: 'return' }),
      });
      if (!response.ok) {
        const payload = (await response.json()) as { data?: { refresh?: { error?: string } } };
        throw new Error(payload.data?.refresh?.error || 'Unable to refresh signal');
      }
      startTransition(() => {
        router.refresh();
      });
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : 'Unable to refresh signal');
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="space-y-2">
      <Button
        type="button"
        variant="secondary"
        onClick={handleRefresh}
        disabled={isPending || isRefreshing}
        className="w-full justify-center"
      >
        <RefreshCcw className={`mr-2 h-4 w-4 ${isPending || isRefreshing ? 'animate-spin' : ''}`} />
        {isPending || isRefreshing ? 'Refreshing...' : 'Refresh OpenBB Data'}
      </Button>
      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  );
}
