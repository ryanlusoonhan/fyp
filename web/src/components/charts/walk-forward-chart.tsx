'use client';

import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface WalkForwardChartPoint {
  windowId: number;
  aiReturnPct: number;
  buyHoldReturnPct: number;
}

interface WalkForwardChartProps {
  data: WalkForwardChartPoint[];
}

export function WalkForwardChart({ data }: WalkForwardChartProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setMounted(true));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  if (!mounted) {
    return <div className="h-80 w-full animate-pulse rounded-xl border border-border/50 bg-panel-strong/50" />;
  }

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 2, bottom: 0 }}>
          <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
          <XAxis dataKey="windowId" stroke="rgba(148,163,184,0.7)" axisLine={false} tickLine={false} />
          <YAxis stroke="rgba(148,163,184,0.7)" axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{
              background: 'rgba(2, 6, 23, 0.95)',
              border: '1px solid rgba(71,85,105,0.6)',
              borderRadius: 12,
            }}
          />
          <Legend />
          <Bar dataKey="aiReturnPct" fill="#f5a524" radius={[6, 6, 0, 0]} name="AI Return %" />
          <Bar dataKey="buyHoldReturnPct" fill="#6fa8ff" radius={[6, 6, 0, 0]} name="B&H Return %" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
