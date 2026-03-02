import { describe, expect, it } from 'vitest';

import { buildRefreshInferenceArgs } from './route';

describe('signal refresh route helpers', () => {
  it('builds refresh inference arguments with openbb live refresh', () => {
    const args = buildRefreshInferenceArgs('return');
    expect(args).toEqual([
      '--objective',
      'return',
      '--json',
      '--refresh-openbb',
      '--refresh-mode',
      'live',
    ]);
  });

  it('builds refresh inference arguments for accuracy objective', () => {
    const args = buildRefreshInferenceArgs('accuracy');
    expect(args).toEqual([
      '--objective',
      'accuracy',
      '--json',
      '--refresh-openbb',
      '--refresh-mode',
      'live',
    ]);
  });
});
