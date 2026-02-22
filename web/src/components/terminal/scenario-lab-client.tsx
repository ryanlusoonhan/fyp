'use client';

import { useState } from 'react';
import { LoaderCircle, Play } from 'lucide-react';

import { ScenarioThresholdChart } from '@/components/charts/scenario-threshold-chart';
import { Button } from '@/components/ui/button';
import type { ScenarioRunResult } from '@/lib/types';

interface ScenarioLabClientProps {
  initialResult: ScenarioRunResult;
}

interface ScenarioRequestPayload {
  objective: 'f1' | 'return';
  thresholdMin: number;
  thresholdMax: number;
  step: number;
  cost: number;
  barrierWindow: number;
}

const DEFAULT_FORM: ScenarioRequestPayload = {
  objective: 'return',
  thresholdMin: 0.3,
  thresholdMax: 0.7,
  step: 0.01,
  cost: 0.001,
  barrierWindow: 10,
};

export function ScenarioLabClient({ initialResult }: ScenarioLabClientProps) {
  const [form, setForm] = useState<ScenarioRequestPayload>(DEFAULT_FORM);
  const [result, setResult] = useState<ScenarioRunResult>(initialResult);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runScenarioRequest() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/pro/scenario/run', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-plan-id': 'elite',
        },
        body: JSON.stringify(form),
      });

      const payload = (await response.json()) as {
        error?: string;
        data?: ScenarioRunResult;
      };

      if (!response.ok || !payload.data) {
        setError(payload.error ?? 'Scenario request failed.');
        return;
      }

      setResult(payload.data);
    } catch {
      setError('Network error while running scenario.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3 rounded-2xl border border-border bg-panel-strong p-4 md:grid-cols-2 xl:grid-cols-3">
        <label className="space-y-1 text-xs text-muted">
          Objective
          <select
            value={form.objective}
            onChange={(event) => setForm((prev) => ({ ...prev, objective: event.target.value as 'f1' | 'return' }))}
            className="h-10 w-full rounded-lg border border-border bg-panel px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-accent/40"
          >
            <option value="return">Return</option>
            <option value="f1">F1</option>
          </select>
        </label>
        <label className="space-y-1 text-xs text-muted">
          Threshold Min
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={form.thresholdMin}
            onChange={(event) => setForm((prev) => ({ ...prev, thresholdMin: Number(event.target.value) }))}
            className="h-10 w-full rounded-lg border border-border bg-panel px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-accent/40"
          />
        </label>
        <label className="space-y-1 text-xs text-muted">
          Threshold Max
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={form.thresholdMax}
            onChange={(event) => setForm((prev) => ({ ...prev, thresholdMax: Number(event.target.value) }))}
            className="h-10 w-full rounded-lg border border-border bg-panel px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-accent/40"
          />
        </label>
        <label className="space-y-1 text-xs text-muted">
          Step
          <input
            type="number"
            step="0.005"
            min="0.005"
            max="0.2"
            value={form.step}
            onChange={(event) => setForm((prev) => ({ ...prev, step: Number(event.target.value) }))}
            className="h-10 w-full rounded-lg border border-border bg-panel px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-accent/40"
          />
        </label>
        <label className="space-y-1 text-xs text-muted">
          Cost
          <input
            type="number"
            step="0.0005"
            min="0"
            max="0.02"
            value={form.cost}
            onChange={(event) => setForm((prev) => ({ ...prev, cost: Number(event.target.value) }))}
            className="h-10 w-full rounded-lg border border-border bg-panel px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-accent/40"
          />
        </label>
        <label className="space-y-1 text-xs text-muted">
          Barrier Window
          <input
            type="number"
            step="1"
            min="1"
            max="30"
            value={form.barrierWindow}
            onChange={(event) => setForm((prev) => ({ ...prev, barrierWindow: Number(event.target.value) }))}
            className="h-10 w-full rounded-lg border border-border bg-panel px-3 text-sm text-foreground outline-none focus:ring-2 focus:ring-accent/40"
          />
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={runScenarioRequest} disabled={loading}>
          {loading ? <LoaderCircle className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
          Run
        </Button>
        {error ? <p className="text-sm text-rose-300">{error}</p> : null}
      </div>

      <div className="rounded-2xl border border-border bg-panel-strong p-4">
        <p className="text-[11px] uppercase tracking-[0.14em] text-muted">Best Threshold</p>
        <p className="mt-1 font-display text-4xl">{result.bestThreshold.toFixed(2)}</p>
        <p className="text-sm text-slate-300">Best score: {result.bestScore.toFixed(4)} ({result.objective})</p>
      </div>

      <ScenarioThresholdChart data={result.candidates} />
    </div>
  );
}
