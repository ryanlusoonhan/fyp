import { NextResponse } from 'next/server';

import { getLatestSignal, refreshLatestSignal } from '@/lib/data/signal-repository';
import type { Objective } from '@/lib/types';

export function buildRefreshInferenceArgs(objective: Objective = 'accuracy'): string[] {
  return ['--objective', objective, '--json', '--refresh-openbb', '--refresh-mode', 'live'];
}

function normalizeObjective(value: unknown): Objective {
  if (value === 'accuracy') {
    return 'accuracy';
  }
  return value === 'f1' ? 'f1' : 'return';
}

export async function POST(request: Request) {
  let objective: Objective = 'accuracy';
  try {
    const body = (await request.json()) as { objective?: Objective };
    objective = normalizeObjective(body?.objective);
  } catch {
    objective = 'accuracy';
  }

  try {
    const signal = await refreshLatestSignal(objective);
    return NextResponse.json({
      data: {
        signal,
        refresh: {
          commandArgs: buildRefreshInferenceArgs(objective),
        },
      },
    });
  } catch (error) {
    const fallback = await getLatestSignal(objective);
    return NextResponse.json(
      {
        data: {
          signal: fallback,
          refresh: {
            commandArgs: buildRefreshInferenceArgs(objective),
            error: error instanceof Error ? error.message : 'Refresh failed',
          },
        },
      },
      { status: 500 },
    );
  }
}
