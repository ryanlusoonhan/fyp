export type Classification = 'BUY' | 'NO_BUY';
export type Objective = 'f1' | 'return';
export type PlanId = 'free' | 'pro' | 'elite';

export interface Signal {
  id: string;
  asOfDate: string;
  market: 'HSI';
  predictedClass: 0 | 1;
  classification: Classification;
  probBuy: number;
  probNoBuy: number;
  threshold: number;
  objective: Objective;
  modelVersion: string;
  confidenceBand: 'low' | 'medium' | 'high';
  dataStatus?: 'fresh' | 'stale' | 'empty' | string;
  lastRefreshAt?: string;
  latestMarketDate?: string;
  dataProvider?: string;
}

export interface SignalExplanation {
  signalId: string;
  confidenceBand: 'low' | 'medium' | 'high';
  keyDrivers: Array<{
    name: string;
    contribution: number;
    direction: 'up' | 'down';
    narrative: string;
  }>;
  regimeTag: 'RiskOn' | 'RiskOff' | 'Neutral';
  invalidationTriggers: string[];
  thesisSummary: string;
}

export interface WalkForwardWindow {
  windowId: number;
  startIdx: number;
  endIdx: number;
  nSamples: number;
  accuracy: number;
  f1: number;
  aiReturnPct: number;
  buyHoldReturnPct: number;
}

export interface PerformanceSummary {
  latestAccuracy: number;
  latestF1: number;
  latestAiReturnPct: number;
  latestBuyHoldReturnPct: number;
  meanAccuracy: number;
  meanF1: number;
  meanAiReturnPct: number;
  meanBuyHoldReturnPct: number;
}

export interface ScenarioRunInput {
  objective: Objective;
  thresholdMin: number;
  thresholdMax: number;
  step: number;
  cost: number;
  barrierWindow: number;
}

export interface ScenarioRunResult {
  objective: Objective;
  bestThreshold: number;
  bestScore: number;
  candidates: Array<{
    threshold: number;
    score: number;
    buyRate: number;
  }>;
}

export interface PricingTier {
  id: PlanId;
  name: string;
  priceMonthlyUsd: number;
  description: string;
  cta: string;
  highlighted?: boolean;
  features: string[];
}
