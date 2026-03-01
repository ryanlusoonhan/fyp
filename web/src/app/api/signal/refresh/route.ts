import { NextResponse } from 'next/server';

import { getRequestContext } from '@/lib/api/auth';
import { getLatestSignal, refreshLatestSignal } from '@/lib/data/signal-repository';
import { canAccessFeature } from '@/lib/domain/entitlements';
import type { Objective } from '@/lib/types';

export function buildRefreshInferenceArgs(objective: Objective = 'return'): string[] {
  return ['--objective', objective, '--json', '--refresh-openbb', '--refresh-mode', 'live'];
}

function normalizeObjective(value: unknown): Objective {
  return value === 'f1' ? 'f1' : 'return';
}

export async function POST(request: Request) {
  let objective: Objective = 'return';
  try {
    const body = (await request.json()) as { objective?: Objective };
    objective = normalizeObjective(body?.objective);
  } catch {
    objective = 'return';
  }

  const auth = await getRequestContext();
  if (!canAccessFeature(auth.plan, 'latest-signal-live')) {
    return NextResponse.json({ error: 'Upgrade required for live refresh.' }, { status: 403 });
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
