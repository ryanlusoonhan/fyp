import { describe, expect, it } from 'vitest';

import { normalizeObjective } from '@/lib/services/scenario-service';

describe('scenario-service objective normalization', () => {
  it('keeps accuracy objective when requested', () => {
    expect(normalizeObjective('accuracy')).toBe('accuracy');
  });

  it('falls back unknown objective to return', () => {
    expect(normalizeObjective('unknown')).toBe('return');
  });
});

