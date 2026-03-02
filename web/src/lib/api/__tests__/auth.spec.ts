import { describe, expect, it } from 'vitest';

import { parseBearerAuthHeader } from '@/lib/api/auth';

describe('auth header parsing', () => {
  it('extracts bearer token', () => {
    expect(parseBearerAuthHeader('Bearer abc123')).toBe('abc123');
    expect(parseBearerAuthHeader('bearer xyz')).toBe('xyz');
  });

  it('returns null for invalid authorization headers', () => {
    expect(parseBearerAuthHeader('Basic abc123')).toBeNull();
    expect(parseBearerAuthHeader('Bearer')).toBeNull();
    expect(parseBearerAuthHeader('')).toBeNull();
    expect(parseBearerAuthHeader(null)).toBeNull();
  });
});
