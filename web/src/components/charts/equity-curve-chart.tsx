'use client';

import { useEffect, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface Point {
  label: string;
  ai: number;
  benchmark: number;
}

interface EquityCurveChartProps {
  data: Point[];
}

export function EquityCurveChart({ data }: EquityCurveChartProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setMounted(true));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  if (!mounted) {
    return <div className="h-72 w-full animate-pulse rounded-xl border border-border/50 bg-panel-strong/50" />;
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ left: 4, right: 10, top: 12, bottom: 8 }}>
          <defs>
            <linearGradient id="aiFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(245, 165, 36, 0.82)" />
              <stop offset="100%" stopColor="rgba(245, 165, 36, 0.04)" />
            </linearGradient>
            <linearGradient id="benchmarkFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(111, 168, 255, 0.45)" />
              <stop offset="100%" stopColor="rgba(111, 168, 255, 0.02)" />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148,163,184,0.18)" vertical={false} />
          <XAxis dataKey="label" stroke="rgba(148,163,184,0.7)" tickLine={false} axisLine={false} />
          <YAxis
            stroke="rgba(148,163,184,0.7)"
            tickLine={false}
            axisLine={false}
            width={56}
            tickFormatter={(value) => `${Math.round(value / 1000)}k`}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(2, 6, 23, 0.95)',
              border: '1px solid rgba(71,85,105,0.6)',
              borderRadius: 12,
            }}
          />
          <Area type="monotone" dataKey="benchmark" stroke="#6fa8ff" fill="url(#benchmarkFill)" strokeWidth={2} />
          <Area type="monotone" dataKey="ai" stroke="#f5a524" fill="url(#aiFill)" strokeWidth={2.5} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
