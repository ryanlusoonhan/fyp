import Link from 'next/link';

import { Button } from '@/components/ui/button';
import { Card, CardDescription, CardTitle } from '@/components/ui/card';

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <Card className="w-full max-w-xl border-accent/30 text-center">
        <p className="text-xs uppercase tracking-[0.12em] text-muted">404</p>
        <CardTitle className="mt-3 text-4xl">Signal Not Found</CardTitle>
        <CardDescription className="mt-2">
          The page you requested does not exist in this terminal snapshot.
        </CardDescription>
        <div className="mt-6 flex justify-center gap-3">
          <Button asChild variant="secondary">
            <Link href="/">Back Home</Link>
          </Button>
          <Button asChild>
            <Link href="/dashboard">Open Dashboard</Link>
          </Button>
        </div>
      </Card>
    </div>
  );
}
