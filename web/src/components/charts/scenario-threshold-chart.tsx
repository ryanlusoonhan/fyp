'use client';

import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface ScenarioThresholdPoint {
  threshold: number;
  score: number;
}

interface ScenarioThresholdChartProps {
  data: ScenarioThresholdPoint[];
}

export function ScenarioThresholdChart({ data }: ScenarioThresholdChartProps) {
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
        <LineChart data={data} margin={{ top: 4, right: 10, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
          <XAxis
            dataKey="threshold"
            stroke="rgba(148,163,184,0.75)"
            axisLine={false}
            tickLine={false}
            tickFormatter={(value) => Number(value).toFixed(2)}
          />
          <YAxis stroke="rgba(148,163,184,0.75)" axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(value: number | undefined) => (typeof value === 'number' ? value.toFixed(4) : '0.0000')}
            contentStyle={{
              background: 'rgba(2, 6, 23, 0.95)',
              border: '1px solid rgba(71,85,105,0.6)',
              borderRadius: 12,
            }}
          />
          <Line type="monotone" dataKey="score" stroke="#f5a524" strokeWidth={2.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
