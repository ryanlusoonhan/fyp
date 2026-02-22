import { describe, expect, it } from 'vitest';

import {
  canAccessFeature,
  type FeatureKey,
} from '@/lib/domain/entitlements';
import type { PlanId } from '@/lib/types';

const FEATURES: FeatureKey[] = [
  'latest-signal-live',
  'signal-history-full',
  'advanced-explainability',
  'scenario-lab',
  'walk-forward-deep-dive',
  'report-export',
];

describe('entitlements', () => {
  it('keeps free users restricted', () => {
    const plan: PlanId = 'free';
    expect(canAccessFeature(plan, 'latest-signal-live')).toBe(false);
    expect(canAccessFeature(plan, 'signal-history-full')).toBe(false);
    expect(canAccessFeature(plan, 'advanced-explainability')).toBe(false);
  });

  it('allows pro features and blocks elite-only features', () => {
    const plan: PlanId = 'pro';
    expect(canAccessFeature(plan, 'latest-signal-live')).toBe(true);
    expect(canAccessFeature(plan, 'advanced-explainability')).toBe(true);
    expect(canAccessFeature(plan, 'scenario-lab')).toBe(false);
    expect(canAccessFeature(plan, 'report-export')).toBe(false);
  });

  it('unlocks all premium features for elite', () => {
    const plan: PlanId = 'elite';
    for (const feature of FEATURES) {
      expect(canAccessFeature(plan, feature)).toBe(true);
    }
  });
});
