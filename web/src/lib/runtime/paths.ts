import path from 'node:path';

export function getRepoRoot(): string {
  const configured = process.env.NELL_REPO_ROOT;
  if (configured && configured.trim().length > 0) {
    return configured.trim();
  }
  // process.cwd() is /.../fyp/web when running Next from this app.
  return path.resolve(process.cwd(), '..');
}

export function getModelsDir(): string {
  return path.resolve(getRepoRoot(), 'models');
}

export function getProcessedDataDir(): string {
  return path.resolve(getRepoRoot(), 'data', 'processed');
}
