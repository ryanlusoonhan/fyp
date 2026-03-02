import { NextResponse } from 'next/server';
import { z } from 'zod';

import { normalizeObjective, runScenario } from '@/lib/services/scenario-service';

const bodySchema = z.object({
  objective: z.enum(['accuracy', 'f1', 'return']).optional(),
  thresholdMin: z.number().min(0).max(1).default(0.3),
  thresholdMax: z.number().min(0).max(1).default(0.7),
  step: z.number().positive().max(0.2).default(0.01),
  cost: z.number().min(0).max(0.02).default(0.001),
  barrierWindow: z.number().int().min(1).max(30).default(10),
});

export async function POST(request: Request) {
  const payload = await request.json().catch(() => ({}));
  const parsed = bodySchema.safeParse(payload);

  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid scenario payload.', details: parsed.error.flatten() }, { status: 400 });
  }

  const result = await runScenario({
    ...parsed.data,
    objective: normalizeObjective(parsed.data.objective),
  });

  return NextResponse.json({ data: result });
}
