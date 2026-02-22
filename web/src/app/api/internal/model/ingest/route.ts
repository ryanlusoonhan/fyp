import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

import { getLatestSignal } from '@/lib/data/signal-repository';
import { getPerformanceSummary, getWalkForwardWindows } from '@/lib/data/walk-forward-repository';

function readEnv(name: string): string | null {
  const value = process.env[name];
  if (!value || value.trim().length === 0) {
    return null;
  }
  return value.trim();
}

export async function POST(request: Request) {
  const authHeader = request.headers.get('authorization');
  const expected = process.env.INTERNAL_INGEST_TOKEN;

  if (!expected || authHeader !== `Bearer ${expected}`) {
    return NextResponse.json({ error: 'Unauthorized ingest request.' }, { status: 401 });
  }

  const [signal, summary, windows] = await Promise.all([
    getLatestSignal('return'),
    getPerformanceSummary(),
    getWalkForwardWindows(),
  ]);

  const supabaseUrl = readEnv('NEXT_PUBLIC_SUPABASE_URL');
  const serviceRole = readEnv('SUPABASE_SERVICE_ROLE_KEY');

  let persisted = false;
  let persistError: string | null = null;
  if (supabaseUrl && serviceRole) {
    const admin = createClient(supabaseUrl, serviceRole, {
      auth: { persistSession: false, autoRefreshToken: false },
    });

    const upsertPayload = {
      as_of_date: signal.asOfDate,
      market: signal.market,
      objective: signal.objective,
      threshold: signal.threshold,
      probability_buy: signal.probBuy,
      probability_no_buy: signal.probNoBuy,
      predicted_class: signal.predictedClass,
      model_version: signal.modelVersion,
      metadata: {
        confidence_band: signal.confidenceBand,
        summary,
      },
    };

    const result = await admin
      .from('signal_snapshots')
      .upsert(upsertPayload, { onConflict: 'as_of_date,market,objective' });

    if (result.error) {
      persistError = result.error.message;
    } else {
      persisted = true;
    }
  }

  return NextResponse.json({
    data: {
      ingestedAt: new Date().toISOString(),
      signal,
      summary,
      windowsCount: windows.length,
      persisted,
      persistError,
    },
  });
}
