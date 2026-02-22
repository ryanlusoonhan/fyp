import type { ReactNode } from 'react';

import { Card, CardDescription, CardTitle } from '@/components/ui/card';

interface KpiCardProps {
  label: string;
  value: string;
  hint: string;
  icon: ReactNode;
}

export function KpiCard({ label, value, hint, icon }: KpiCardProps) {
  return (
    <Card className="space-y-2">
      <div className="flex items-center justify-between text-muted">
        <p className="text-[11px] uppercase tracking-[0.14em]">{label}</p>
        {icon}
      </div>
      <CardTitle className="text-3xl">{value}</CardTitle>
      <CardDescription className="text-xs">{hint}</CardDescription>
    </Card>
  );
}
