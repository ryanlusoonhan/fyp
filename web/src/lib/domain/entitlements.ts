import type { PlanId } from '@/lib/types';

export type FeatureKey =
  | 'latest-signal-live'
  | 'signal-history-full'
  | 'advanced-explainability'
  | 'scenario-lab'
  | 'walk-forward-deep-dive'
  | 'report-export';

const ENTITLEMENTS: Record<PlanId, Set<FeatureKey>> = {
  free: new Set(),
  pro: new Set(['latest-signal-live', 'signal-history-full', 'advanced-explainability']),
  elite: new Set([
    'latest-signal-live',
    'signal-history-full',
    'advanced-explainability',
    'scenario-lab',
    'walk-forward-deep-dive',
    'report-export',
  ]),
};

export function canAccessFeature(plan: PlanId, feature: FeatureKey): boolean {
  return ENTITLEMENTS[plan].has(feature);
}
