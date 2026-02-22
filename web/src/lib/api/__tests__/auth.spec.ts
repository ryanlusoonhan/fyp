import { describe, expect, it } from 'vitest';

import { parsePlanFromDevToken } from '@/lib/api/auth';

describe('auth debug token parsing', () => {
  it('accepts plan-prefixed debug tokens', () => {
    expect(parsePlanFromDevToken('plan:pro')).toBe('pro');
    expect(parsePlanFromDevToken('plan:elite')).toBe('elite');
  });

  it('accepts demo-prefixed debug tokens', () => {
    expect(parsePlanFromDevToken('demo_free')).toBe('free');
    expect(parsePlanFromDevToken('demo_pro')).toBe('pro');
  });

  it('returns null for unsupported tokens', () => {
    expect(parsePlanFromDevToken('demo_enterprise')).toBeNull();
    expect(parsePlanFromDevToken('Bearer abc')).toBeNull();
    expect(parsePlanFromDevToken(null)).toBeNull();
  });
});
