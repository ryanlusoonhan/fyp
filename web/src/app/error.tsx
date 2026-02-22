'use client';

import { useEffect } from 'react';
import Link from 'next/link';

import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <Card className="w-full max-w-xl border-rose-400/35">
        <p className="text-xs uppercase tracking-[0.12em] text-rose-200">Runtime Exception</p>
        <CardTitle className="mt-3 text-3xl">Interface Session Interrupted</CardTitle>
        <CardDescription className="mt-2">
          We hit an unexpected error while rendering this view. You can retry immediately or return to a stable page.
        </CardDescription>
        <div className="mt-6 flex flex-wrap gap-3">
          <Button variant="secondary" onClick={reset}>
            Retry
          </Button>
          <Button asChild>
            <Link href="/dashboard">Go to Dashboard</Link>
          </Button>
        </div>
      </Card>
    </div>
  );
}
