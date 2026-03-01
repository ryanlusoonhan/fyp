import fs from 'node:fs/promises';
import path from 'node:path';

import { getRepoRoot } from '@/lib/runtime/paths';

interface OpenbbStatusFile {
  provider?: string;
  mode?: 'batch' | 'live' | string;
  start_date?: string;
  end_date?: string;
  rows_index_history?: number;
  rows_equity_history?: number;
  rows_currency_history?: number;
  rows_news_recent?: number;
  rows_training?: number;
  last_refresh_at?: string;
  latest_market_date?: string;
  data_status?: string;
  training_file?: string | null;
}

interface OpenbbManifestFile {
  provider?: string;
  index_symbols?: string[];
  equity_symbols?: string[];
  currency_symbols?: string[];
}

interface WeeklyModelConfigFile {
  data_source?: string;
  data_file?: string;
  feature_set_version?: string;
  feature_cols?: string[];
}

export interface OpenbbOverview {
  source: {
    provider: string | null;
    mode: string | null;
    status: string | null;
    lastRefreshAt: string | null;
    latestMarketDate: string | null;
  };
  snapshots: Array<{
    name: string;
    path: string;
    rows: number | null;
  }>;
  symbols: {
    index: string[];
    equity: string[];
    currency: string[];
  };
  modelIntegration: {
    enabled: boolean;
    dataSource: string | null;
    dataFile: string | null;
    featureSetVersion: string | null;
    featureCount: number;
    openbbFeatureCount: number;
  };
}

async function readJsonSafe<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function countCsvRows(filePath: string): Promise<number | null> {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    const rowCount = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0).length;

    if (rowCount <= 1) {
      return 0;
    }
    return rowCount - 1;
  } catch {
    return null;
  }
}

function countOpenbbFeatures(featureCols: string[]): number {
  const prefixes = ['VIX_', 'TNX_', 'GSPC_', 'USDHKD_', 'USDCNY_', 'HK_Breadth_'];
  return featureCols.filter((col) => prefixes.some((prefix) => col.startsWith(prefix))).length;
}

export async function getOpenbbOverview(): Promise<OpenbbOverview> {
  const repoRoot = getRepoRoot();
  const statusPath = path.resolve(repoRoot, 'models', 'openbb_refresh_status.json');
  const manifestPath = path.resolve(repoRoot, 'data', 'raw', 'openbb', 'manifest.json');
  const modelConfigPath = path.resolve(repoRoot, 'models', 'model_config_weekly.json');

  const [status, manifest, modelConfig] = await Promise.all([
    readJsonSafe<OpenbbStatusFile>(statusPath),
    readJsonSafe<OpenbbManifestFile>(manifestPath),
    readJsonSafe<WeeklyModelConfigFile>(modelConfigPath),
  ]);

  const snapshots = [
    { name: 'Index History', path: path.resolve(repoRoot, 'data', 'raw', 'openbb', 'index_history.csv') },
    { name: 'Equity History', path: path.resolve(repoRoot, 'data', 'raw', 'openbb', 'equity_history.csv') },
    { name: 'Currency History', path: path.resolve(repoRoot, 'data', 'raw', 'openbb', 'currency_history.csv') },
    { name: 'Recent News', path: path.resolve(repoRoot, 'data', 'raw', 'openbb', 'news_recent.csv') },
    { name: 'Training Dataset', path: path.resolve(repoRoot, 'data', 'processed', 'training_data_openbb.csv') },
  ];

  const snapshotRows = await Promise.all(
    snapshots.map(async (snapshot) => ({
      name: snapshot.name,
      path: path.relative(repoRoot, snapshot.path),
      rows: await countCsvRows(snapshot.path),
    })),
  );

  const featureCols = modelConfig?.feature_cols ?? [];
  const openbbFeatureCount = countOpenbbFeatures(featureCols);
  const dataSource = modelConfig?.data_source ?? null;
  const dataFile = modelConfig?.data_file ?? null;
  const enabled =
    typeof dataSource === 'string' &&
    dataSource.includes('openbb') &&
    typeof dataFile === 'string' &&
    dataFile.includes('training_data_openbb.csv');

  return {
    source: {
      provider: status?.provider ?? manifest?.provider ?? null,
      mode: status?.mode ?? null,
      status: status?.data_status ?? null,
      lastRefreshAt: status?.last_refresh_at ?? null,
      latestMarketDate: status?.latest_market_date ?? null,
    },
    snapshots: snapshotRows,
    symbols: {
      index: manifest?.index_symbols ?? [],
      equity: manifest?.equity_symbols ?? [],
      currency: manifest?.currency_symbols ?? [],
    },
    modelIntegration: {
      enabled,
      dataSource,
      dataFile,
      featureSetVersion: modelConfig?.feature_set_version ?? null,
      featureCount: featureCols.length,
      openbbFeatureCount,
    },
  };
}
