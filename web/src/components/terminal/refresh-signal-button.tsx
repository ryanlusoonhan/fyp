'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { RefreshCcw } from 'lucide-react';

import { Button } from '@/components/ui/button';

export function RefreshSignalButton() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  const handleRefresh = async () => {
    setError(null);
    try {
      const response = await fetch('/api/signal/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
    }
  };

  return (
    <div className="space-y-2">
      <Button type="button" variant="secondary" onClick={handleRefresh} disabled={isPending} className="w-full justify-center">
        <RefreshCcw className={`mr-2 h-4 w-4 ${isPending ? 'animate-spin' : ''}`} />
        {isPending ? 'Refreshing...' : 'Refresh OpenBB Data'}
      </Button>
      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  );
}
