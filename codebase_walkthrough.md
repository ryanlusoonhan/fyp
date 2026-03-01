# FYP Codebase Deep Dive Report

Updated: 2026-03-01

This report is written to make the repository understandable again, especially for someone who did not make all recent changes.

## 1) Quick orientation (what is actually active right now)

There are two pipelines in this repo:

1. Current active pipeline (weekly + OpenBB)
   - `openbb_refresh.py` -> fetches market data with OpenBB
   - `train_weekly.py` -> trains weekly binary model (BUY/NO_BUY)
   - `backtest_weekly.py` -> evaluates and writes walk-forward metrics
   - `weekly_inference.py` -> generates latest weekly signal (used by web app)
   - `scenario_weekly.py` -> runs threshold scenario analysis (used by scenario page/API)
   - `web/` -> full-stack dashboard that reads Python outputs

2. Legacy pipeline (older daily/sentiment path, mostly kept for reference)
   - `train.py`, `backtest.py`, `inference.py`
   - `data/processed/training_data.csv`, `models/model_config.json`, `models/best_model.pth`

If Nell only wants the weekly path, he can mostly ignore the legacy daily files.

## 2) Status labels used below

- ACTIVE: part of the current weekly + OpenBB workflow
- SUPPORTING: helper/config/test/doc file used around active flow
- LEGACY: old approach retained for fallback/history
- GENERATED: output artifact produced by scripts; usually not hand-edited
- LOCAL: machine-local file, cache, or environment file

## 2.1) Machine-local trees intentionally not expanded file-by-file

These exist in the folder but are not "project logic" and are intentionally summarized, not individually documented:

- `.venv/` (LOCAL)
  - Python virtual environment binaries and installed packages.
  - Recreated by `python -m venv` + `pip install -r requirements.txt`.
- `__pycache__/`, `src/__pycache__/`, `tests/__pycache__/` (LOCAL GENERATED)
  - Python bytecode caches.
- `.ruff_cache/` (LOCAL GENERATED)
  - Ruff linter cache.
- `web/node_modules/` (LOCAL GENERATED)
  - NPM installed dependencies.
- `web/.next/` (LOCAL GENERATED)
  - Next.js build/dev artifacts.

## 3) Root folder file-by-file

### `/Users/ryanlu/Projects/Nell/fyp/.gitignore` (SUPPORTING)
- Main ignore policy.
- Excludes local env files, caches, web build outputs, and generated OpenBB/model artifacts.
- Purpose: prevent accidentally committing heavy/generated files.

### `/Users/ryanlu/Projects/Nell/fyp/.dockerignore` (SUPPORTING)
- Controls what goes into Docker build context.
- Excludes `.git`, local virtualenv, pycache, web build outputs, models, and OpenBB raw files for smaller/faster build.

### `/Users/ryanlu/Projects/Nell/fyp/README.md` (ACTIVE)
- Short Docker-first run instructions.
- Current "how to run" starting point.

### `/Users/ryanlu/Projects/Nell/fyp/Makefile` (SUPPORTING)
- Convenience wrappers for Docker commands:
  - build/up/down/shell
  - refresh/train/backtest/signal
  - Python tests and web tests

### `/Users/ryanlu/Projects/Nell/fyp/requirements.txt` (ACTIVE)
- Python dependency pin set.
- Includes model stack (`torch`, `scikit-learn`, `matplotlib`, `joblib`) and OpenBB stack (`openbb`, `openbb-yfinance`).

### `/Users/ryanlu/Projects/Nell/fyp/docker-compose.yml` (ACTIVE)
- One service: `app`.
- Maps port `3000:3000`.
- Mounts repo into container.
- Sets environment (`PYTHON_BIN`, `NELL_REPO_ROOT`, `NELL_QUIET_DEVICE`).
- Entrypoint uses `/usr/local/bin/start-web`.

### `/Users/ryanlu/Projects/Nell/fyp/openbb_refresh.py` (ACTIVE)
- CLI wrapper around `src/data/openbb_ingestion.py`.
- Supports `--mode batch|live`, date range/lookback, and optional training dataset write.

### `/Users/ryanlu/Projects/Nell/fyp/train_weekly.py` (ACTIVE)
- Weekly training script (binary classifier).
- Reads OpenBB dataset by default; can fallback to legacy training file if needed.
- Engineers market-only features.
- Builds triple-barrier labels.
- Splits data time-safely with gap.
- Scales features and trains LSTM with class weights and early stopping.
- Writes:
  - `data/processed/scalers/feature_scaler_weekly.pkl`
  - `models/best_model_weekly_binary.pth`
  - `models/model_config_weekly.json`
  - `models/training_history_weekly_binary.png`

### `/Users/ryanlu/Projects/Nell/fyp/backtest_weekly.py` (ACTIVE)
- Weekly backtest and threshold optimization.
- Supports objective-based threshold tuning (`f1` or `return`).
- Computes walk-forward windows and latest-window metrics.
- Writes:
  - `models/backtest_weekly_walk_forward.csv`
  - `models/backtest_limited_<N>.png`

### `/Users/ryanlu/Projects/Nell/fyp/weekly_inference.py` (ACTIVE)
- Main inference entrypoint used by web server.
- Loads weekly config/scaler/model.
- Can refresh OpenBB data before inference (`--refresh-openbb`).
- Can auto-infer threshold from validation when not manually provided.
- Outputs JSON payload for API/web.
- Upserts inference history in `models/signal_history_weekly.csv` (unless `--no-append-history`).

### `/Users/ryanlu/Projects/Nell/fyp/scenario_weekly.py` (ACTIVE)
- Runs threshold grid search scenario on validation set.
- Used by web scenario API/page through Python subprocess execution.
- Output is JSON with best threshold + candidate curve.

### `/Users/ryanlu/Projects/Nell/fyp/train.py` (LEGACY)
- Older daily directional training pipeline using sentiment features.
- Writes legacy model artifacts (`models/best_model.pth`, `models/model_config.json`).

### `/Users/ryanlu/Projects/Nell/fyp/backtest.py` (LEGACY)
- Older daily backtest for legacy model.
- Uses fixed threshold, writes `models/backtest_result.png`.

### `/Users/ryanlu/Projects/Nell/fyp/inference.py` (LEGACY)
- Older daily signal inference script for legacy model config/artifacts.

### `/Users/ryanlu/Projects/Nell/fyp/structure.txt` (LEGACY DOC)
- Old static tree snapshot; no runtime effect.

### `/Users/ryanlu/Projects/Nell/fyp/CODEBASE_DEEP_DIVE_REPORT.md` (SUPPORTING)
- This report.

## 4) Docker folder

### `/Users/ryanlu/Projects/Nell/fyp/docker/Dockerfile` (ACTIVE)
- Base image: `node:20-bookworm-slim`.
- Installs Python + build tools.
- Creates virtualenv at `/opt/venv`.
- Installs Python deps from root `requirements.txt`.
- Installs web deps from `web/package-lock.json`.
- Copies startup script to `/usr/local/bin/start-web`.

### `/Users/ryanlu/Projects/Nell/fyp/scripts/docker/start-web.sh` (ACTIVE)
- Container startup script.
- Ensures web dependencies exist.
- Runs `npm run dev` on `0.0.0.0:3000`.

## 5) Python source package (`src/`)

### `/Users/ryanlu/Projects/Nell/fyp/src/config.py` (ACTIVE)
- Central constants:
  - device selection (`cuda`/`mps`/`cpu`)
  - model hyperparameters
  - data/model/scaler paths
  - OpenBB symbol universe and artifact file paths

### `/Users/ryanlu/Projects/Nell/fyp/src/utils.py` (ACTIVE + LEGACY support)
- Feature engineering:
  - `engineer_features_market_only` (ACTIVE weekly)
  - `engineer_features_past_only` (LEGACY sentiment)
- Labeling:
  - `add_triple_barrier_labels` (weekly target generation)
- Sequence and split helpers:
  - `create_sequences`
  - `time_split_with_gap`
- Plot helper:
  - `plot_training_loss`

### `/Users/ryanlu/Projects/Nell/fyp/src/model.py` (ACTIVE + LEGACY support)
- Defines:
  - `LSTMModel` (ACTIVE classifier used now)
  - `TransformerModel`, `HybridModel`, `DirectionalLoss` (not used in current weekly path)
- Current runtime only needs `LSTMModel`.

### `/Users/ryanlu/Projects/Nell/fyp/src/dataset.py` (ACTIVE + LEGACY support)
- `StockDataset` (ACTIVE classification dataset class).
- `TimeSeriesDataset` (older generic sequence dataset; mostly legacy).

### `/Users/ryanlu/Projects/Nell/fyp/src/data/__init__.py` (SUPPORTING)
- Package marker for `src.data`.

### `/Users/ryanlu/Projects/Nell/fyp/src/data/openbb_client.py` (ACTIVE)
- OpenBB SDK wrapper with provider pinning and retry.
- Normalizes OpenBB response shapes for:
  - index/equity/currency historical data
  - recent company news
- Raises typed errors for clear failure handling.

### `/Users/ryanlu/Projects/Nell/fyp/src/data/openbb_ingestion.py` (ACTIVE)
- End-to-end OpenBB ingestion logic:
  - resolves refresh window (batch/live)
  - pulls index/equity/currency/news
  - constructs canonical training dataset (HSI base + exogenous data + breadth features)
  - writes raw snapshots, manifest, refresh status, processed training dataset
- Exposes:
  - `run_openbb_refresh`
  - `load_refresh_status`

## 6) Data folder (`data/`)

### `data/raw/` files

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/hsi_price_history.csv` (LEGACY INPUT)
- Old price history source used in earlier pipeline/notebooks.
- Note: first row appears malformed (header-like values in data row), so this file should not be trusted for new workflow.

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/scraped_news_dump.csv` (LEGACY INPUT)
- Historical scraped headlines (`Date`, `Headline`) used for earlier sentiment workflow.

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/openbb/index_history.csv` (GENERATED ACTIVE)
- Latest OpenBB index snapshot for configured index symbols (`^HSI`, `^VIX`, `^TNX`, `^GSPC`).

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/openbb/equity_history.csv` (GENERATED ACTIVE)
- Latest OpenBB snapshot for HK equity breadth basket.

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/openbb/currency_history.csv` (GENERATED ACTIVE)
- Latest OpenBB snapshot for FX proxies (`USDHKD`, `USDCNY`).

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/openbb/news_recent.csv` (GENERATED ACTIVE CONTEXT)
- Recent news rows from OpenBB provider.
- Used as context/visibility; not required as historical training feature in current model.

#### `/Users/ryanlu/Projects/Nell/fyp/data/raw/openbb/manifest.json` (GENERATED ACTIVE)
- Metadata for last refresh, symbol universe, provider.
- Used by `/openbb` dashboard page.

### `data/processed/` files

#### `/Users/ryanlu/Projects/Nell/fyp/data/processed/training_data_openbb.csv` (GENERATED ACTIVE)
- Main processed dataset for weekly OpenBB pipeline.
- Contains HSI OHLCV + macro + FX + breadth + base return/target columns.

#### `/Users/ryanlu/Projects/Nell/fyp/data/processed/training_data.csv` (LEGACY INPUT)
- Old merged training data with sentiment columns.
- Retained as fallback reference.

#### `/Users/ryanlu/Projects/Nell/fyp/data/processed/daily_sentiment.csv` (LEGACY INPUT)
- Sentiment score time series from previous approach.

### `data/processed/scalers/` files

#### `/Users/ryanlu/Projects/Nell/fyp/data/processed/scalers/feature_scaler_weekly.pkl` (GENERATED ACTIVE)
- StandardScaler fitted for active weekly feature columns.

#### `/Users/ryanlu/Projects/Nell/fyp/data/processed/scalers/feature_scaler.pkl` (GENERATED LEGACY)
- Legacy scaler for old daily model.

#### `/Users/ryanlu/Projects/Nell/fyp/data/processed/scalers/target_scaler.pkl` (GENERATED LEGACY)
- Legacy artifact (regression-era/older flow), not used by current weekly classifier path.

## 7) Models folder (`models/`)

### Config and runtime metadata

#### `/Users/ryanlu/Projects/Nell/fyp/models/model_config_weekly.json` (GENERATED ACTIVE)
- Current weekly model contract:
  - feature list
  - seq length
  - barrier settings
  - data source/file
  - default threshold
- Confirms OpenBB integration via `data_source: openbb_yfinance`.

#### `/Users/ryanlu/Projects/Nell/fyp/models/openbb_refresh_status.json` (GENERATED ACTIVE)
- Latest refresh status consumed by inference and web UI.
- Includes freshness markers.

#### `/Users/ryanlu/Projects/Nell/fyp/models/signal_history_weekly.csv` (GENERATED ACTIVE)
- Persistent signal log appended/upserted by `weekly_inference.py`.
- Used by `web/src/lib/data/signal-repository.ts` for real history.

#### `/Users/ryanlu/Projects/Nell/fyp/models/backtest_weekly_walk_forward.csv` (GENERATED ACTIVE)
- Window-level validation metrics used by walk-forward charts/summary.

### Active weekly model artifacts

#### `/Users/ryanlu/Projects/Nell/fyp/models/best_model_weekly_binary.pth` (GENERATED ACTIVE)
- Trained LSTM state dict for active weekly BUY/NO_BUY model.

#### `/Users/ryanlu/Projects/Nell/fyp/models/training_history_weekly_binary.png` (GENERATED ACTIVE)
- Weekly training loss chart output.

#### `/Users/ryanlu/Projects/Nell/fyp/models/backtest_limited_100.png` (GENERATED ACTIVE)
- Latest backtest equity plot for evaluation window.

### Legacy/experimental artifacts kept in repo folder

#### `/Users/ryanlu/Projects/Nell/fyp/models/model_config.json` (GENERATED LEGACY)
- Legacy config used by `train.py`/`inference.py`.

#### `/Users/ryanlu/Projects/Nell/fyp/models/best_model.pth` (GENERATED LEGACY)
- Legacy daily binary model.

#### `/Users/ryanlu/Projects/Nell/fyp/models/best_model_weekly_3class.pth` (GENERATED LEGACY/EXPERIMENT)
- Older weekly 3-class model artifact (not in current runtime path).

#### `/Users/ryanlu/Projects/Nell/fyp/models/training_history.png` (GENERATED LEGACY)
- Training plot for legacy daily run.

#### `/Users/ryanlu/Projects/Nell/fyp/models/training_history_weekly_3class.png` (GENERATED LEGACY/EXPERIMENT)
- Training plot for older weekly 3-class run.

#### `/Users/ryanlu/Projects/Nell/fyp/models/backtest_result.png` (GENERATED LEGACY)
- Backtest plot from legacy daily backtest.

#### `/Users/ryanlu/Projects/Nell/fyp/models/backtest_weekly_binary.png` (GENERATED LEGACY/EXPERIMENT)
- Older weekly backtest visualization.

#### `/Users/ryanlu/Projects/Nell/fyp/models/backtest_weekly_3class.png` (GENERATED LEGACY/EXPERIMENT)
- Older weekly 3-class backtest visualization.

#### `/Users/ryanlu/Projects/Nell/fyp/models/backtest_weekly_comparison.png` (GENERATED LEGACY/EXPERIMENT)
- Older comparison plot between weekly variants.

## 8) Tests (`tests/`)

### `/Users/ryanlu/Projects/Nell/fyp/tests/test_containerization.py` (ACTIVE)
- Verifies Docker files/service wiring exist.

### `/Users/ryanlu/Projects/Nell/fyp/tests/test_directional_pipeline.py` (ACTIVE)
- Core unit checks for directional pipeline helpers, threshold tuning, dataset selection, and payload compatibility.

### `/Users/ryanlu/Projects/Nell/fyp/tests/test_feature_engineering_openbb.py` (ACTIVE)
- Verifies market-only feature engineering columns and anti-leakage behavior.

### `/Users/ryanlu/Projects/Nell/fyp/tests/test_openbb_client.py` (ACTIVE)
- Verifies OpenBB client normalization, provider forwarding, and retry handling.

## 9) Notebooks (`notebooks/`)

### `/Users/ryanlu/Projects/Nell/fyp/notebooks/scrape_and_explore.ipynb` (LEGACY RESEARCH)
- One-cell notebook script for historical scraping/exploration.

### `/Users/ryanlu/Projects/Nell/fyp/notebooks/sentiment_analysis.ipynb` (LEGACY RESEARCH)
- One-cell notebook for sentiment modeling/experiments.

### `/Users/ryanlu/Projects/Nell/fyp/notebooks/data_merging.ipynb` (LEGACY RESEARCH)
- One-cell notebook for data merging/preprocessing prototypes.

## 10) Docs (`docs/` + `.claude/`)

### `/Users/ryanlu/Projects/Nell/fyp/docs/architecture/2026-02-20-signal-terminal-mvp.md` (SUPPORTING)
- Architecture decision document for MVP topology, data flow, and trade-offs.

### `/Users/ryanlu/Projects/Nell/fyp/docs/plans/2026-02-19-weekly-pipeline-enhancements.md` (SUPPORTING/HISTORICAL)
- Execution plan used to build weekly threshold + inference enhancements.

### `/Users/ryanlu/Projects/Nell/fyp/docs/gtm/pricing-and-go-to-market.md` (SUPPORTING/BUSINESS)
- Monetization and GTM notes; no runtime impact.

### `/Users/ryanlu/Projects/Nell/fyp/.claude/product-marketing-context.md` (SUPPORTING/BUSINESS)
- Product positioning and packaging context.

## 11) Web app (`web/`) deep file map

## 11.1 Web root config files

### `/Users/ryanlu/Projects/Nell/fyp/web/package.json` (ACTIVE)
- NPM scripts and dependencies for Next.js app.

### `/Users/ryanlu/Projects/Nell/fyp/web/package-lock.json` (SUPPORTING)
- Exact dependency lock for reproducible installs.

### `/Users/ryanlu/Projects/Nell/fyp/web/next.config.ts` (SUPPORTING)
- Minimal Next config (Turbopack root).

### `/Users/ryanlu/Projects/Nell/fyp/web/eslint.config.mjs` (SUPPORTING)
- ESLint config based on Next presets.

### `/Users/ryanlu/Projects/Nell/fyp/web/postcss.config.mjs` (SUPPORTING)
- PostCSS/Tailwind integration.

### `/Users/ryanlu/Projects/Nell/fyp/web/tsconfig.json` (SUPPORTING)
- TypeScript compiler settings and path aliases.

### `/Users/ryanlu/Projects/Nell/fyp/web/vitest.config.ts` (SUPPORTING)
- Vitest config for node test environment and alias mapping.

### `/Users/ryanlu/Projects/Nell/fyp/web/README.md` (SUPPORTING)
- Web-specific setup and env variable docs.

### `/Users/ryanlu/Projects/Nell/fyp/web/.env.example` (SUPPORTING)
- Template env file for web app runtime variables.

### `/Users/ryanlu/Projects/Nell/fyp/web/.env.local` (LOCAL)
- Local env values (should not be committed).

### `/Users/ryanlu/Projects/Nell/fyp/web/.gitignore` (SUPPORTING)
- Web-local ignore rules.

### `/Users/ryanlu/Projects/Nell/fyp/web/next-env.d.ts` (GENERATED SUPPORTING)
- Next.js TypeScript helper references.

### `/Users/ryanlu/Projects/Nell/fyp/web/tsconfig.tsbuildinfo` (LOCAL GENERATED)
- TypeScript incremental build cache.

## 11.2 Web app routes/pages

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/layout.tsx` (ACTIVE)
- Global app shell (metadata + font loading + global CSS import).

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/globals.css` (ACTIVE)
- Global theme tokens, base styles, and animation classes.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/page.tsx` (ACTIVE)
- Landing page for Stock Prediction Interface.
- Pulls latest signal + performance summary for hero/kpi content.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/pricing/page.tsx` (ACTIVE)
- Marketing pricing page shell + pricing grid.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/error.tsx` (ACTIVE)
- Global runtime error UI.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/not-found.tsx` (ACTIVE)
- 404 page.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/layout.tsx` (ACTIVE)
- Layout wrapper for dashboard terminal pages.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/dashboard/page.tsx` (ACTIVE)
- Main operational dashboard.
- Shows signal hero, KPIs, equity chart, and data reliability panel.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/signals/page.tsx` (ACTIVE)
- Signal history table page.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/walk-forward/page.tsx` (ACTIVE)
- Walk-forward chart page.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/explainability/page.tsx` (ACTIVE)
- Explainability page with driver stack/invalidation.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/scenario-lab/page.tsx` (ACTIVE)
- Scenario lab page (server side pre-run + client controls).

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/openbb/page.tsx` (ACTIVE)
- OpenBB observability page.
- Shows provider/mode/freshness, artifact row counts, symbol universe, and model integration proof.

## 11.3 Web API routes

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/health/route.ts` (ACTIVE)
- Simple health endpoint.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/public/signal/latest/route.ts` (ACTIVE)
- Free-tier masked public signal preview.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/public/performance/summary/route.ts` (ACTIVE)
- Public performance summary endpoint.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/pro/signals/history/route.ts` (ACTIVE)
- Entitlement-gated full signal history endpoint.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/pro/explanations/[signalId]/route.ts` (ACTIVE)
- Entitlement-gated explanation endpoint for specific signal id.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/pro/scenario/run/route.ts` (ACTIVE)
- Entitlement-gated scenario execution endpoint.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/signal/refresh/route.ts` (ACTIVE)
- Manual live refresh endpoint.
- Calls Python inference with OpenBB refresh args.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/internal/model/ingest/route.ts` (ACTIVE)
- Internal token-protected endpoint for ingesting latest signal + summary into Supabase.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/billing/checkout/route.ts` (ACTIVE)
- Stripe checkout session creation (with safe stub fallback if Stripe not configured).

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/billing/portal/route.ts` (ACTIVE)
- Stripe billing portal session creation for authenticated customer.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/api/signal/refresh/route.spec.ts` (SUPPORTING TEST)
- Unit test for refresh argument contract.

## 11.4 Web components

### UI primitives

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/ui/button.tsx` (ACTIVE)
- Button variant system + optional `asChild` composition.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/ui/card.tsx` (ACTIVE)
- Card container and typography helpers.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/ui/badge.tsx` (ACTIVE)
- Badge variants for status labels.

### Terminal components

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/terminal/terminal-shell.tsx` (ACTIVE)
- Main terminal navigation shell/sidebar/top bar.
- Includes `/openbb` nav entry.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/terminal/terminal-layout-client.tsx` (ACTIVE)
- Client wrapper connecting pathname to terminal shell.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/terminal/signal-hero-card.tsx` (ACTIVE)
- Signal summary hero panel with confidence and data status.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/terminal/kpi-card.tsx` (ACTIVE)
- Reusable KPI card.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/terminal/refresh-signal-button.tsx` (ACTIVE)
- Client-side button to call refresh API and refresh page data.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/terminal/scenario-lab-client.tsx` (ACTIVE)
- Interactive form + chart for scenario API execution.

### Chart components

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/charts/equity-curve-chart.tsx` (ACTIVE)
- Recharts area chart for compounded AI vs benchmark.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/charts/walk-forward-chart.tsx` (ACTIVE)
- Recharts bar chart for window-by-window returns.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/charts/scenario-threshold-chart.tsx` (ACTIVE)
- Recharts line chart for threshold score curve.

### Marketing components

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/marketing/pricing-grid-client.tsx` (ACTIVE)
- Client pricing interval toggle + tier cards.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/marketing/pricing-tier-card.tsx` (ACTIVE)
- Individual tier presentation component.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/components/marketing/checkout-button.tsx` (ACTIVE)
- Frontend call to checkout API and redirect handling.

## 11.5 Web library layer

### Runtime + utility

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/runtime/paths.ts` (ACTIVE)
- Resolves repo root/models/processed paths from web server context.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/utils.ts` (ACTIVE)
- Classname merge + formatting helpers (percent/currency).

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/types.ts` (ACTIVE)
- Shared TypeScript contracts for signals, scenarios, pricing, etc.

### Domain logic

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/domain/signal-engine.ts` (ACTIVE)
- Pure TS signal math: classification, confidence bands, threshold optimization.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/domain/entitlements.ts` (ACTIVE)
- Plan feature gates for free/pro/elite.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/domain/pricing.ts` (ACTIVE)
- Pricing tier definitions and helper getter.

### Data repositories + parsing

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/data/signal-repository.ts` (ACTIVE)
- Bridge between Next and Python inference.
- Runs `weekly_inference.py`, parses payload, maps to UI model.
- Reads real `models/signal_history_weekly.csv` with fallback generation.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/data/walk-forward-repository.ts` (ACTIVE)
- Reads walk-forward CSV, computes summary, has fallback sample windows.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/data/openbb-repository.ts` (ACTIVE)
- Reads OpenBB status/manifest/model config/artifact row counts for `/openbb`.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/ingest/walk-forward-parser.ts` (ACTIVE)
- CSV parser for walk-forward rows.

### Service + integration

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/services/scenario-service.ts` (ACTIVE)
- Runs `scenario_weekly.py` subprocess and maps JSON output to UI contract.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/api/auth.ts` (ACTIVE)
- Request auth context and plan resolution.
- Supports dev override headers/tokens; supports Supabase-based plan lookup.

#### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/billing/stripe.ts` (ACTIVE)
- Stripe client/price-id helpers and app URL resolution logic.

## 11.6 Web tests

### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/api/__tests__/auth.spec.ts` (ACTIVE TEST)
- Tests dev token -> plan parsing.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/billing/__tests__/stripe.spec.ts` (ACTIVE TEST)
- Tests Stripe price ID and app URL resolution behavior.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/data/__tests__/signal-repository.spec.ts` (ACTIVE TEST)
- Tests signal payload parsing and history CSV parsing.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/domain/__tests__/entitlements.spec.ts` (ACTIVE TEST)
- Tests feature access matrix for free/pro/elite.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/domain/__tests__/signal-engine.spec.ts` (ACTIVE TEST)
- Tests classification and threshold optimizer behavior.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/ingest/__tests__/walk-forward-parser.spec.ts` (ACTIVE TEST)
- Tests walk-forward CSV parser output typing/coercion.

## 11.7 Web static assets and DB migration

### `/Users/ryanlu/Projects/Nell/fyp/web/public/next.svg` (SUPPORTING ASSET)
- Placeholder/static SVG.

### `/Users/ryanlu/Projects/Nell/fyp/web/public/vercel.svg` (SUPPORTING ASSET)
- Placeholder/static SVG.

### `/Users/ryanlu/Projects/Nell/fyp/web/public/globe.svg` (SUPPORTING ASSET)
- Placeholder/static SVG.

### `/Users/ryanlu/Projects/Nell/fyp/web/public/window.svg` (SUPPORTING ASSET)
- Placeholder/static SVG.

### `/Users/ryanlu/Projects/Nell/fyp/web/public/file.svg` (SUPPORTING ASSET)
- Placeholder/static SVG.

### `/Users/ryanlu/Projects/Nell/fyp/web/src/app/favicon.ico` (SUPPORTING ASSET)
- Browser favicon.

### `/Users/ryanlu/Projects/Nell/fyp/web/supabase/migrations/20260220_000001_mvp_schema.sql` (ACTIVE SUPPORTING)
- Supabase schema:
  - profiles
  - subscriptions
  - signal_snapshots
  - scenario_runs
  - enums + RLS policies + updated_at trigger

## 12) Redundancy and confusion map (what is safe to ignore for now)

If Nell is focused only on weekly OpenBB system, these are likely "not needed day-to-day":

- Legacy scripts: `train.py`, `backtest.py`, `inference.py`
- Legacy processed files:
  - `data/processed/training_data.csv`
  - `data/processed/daily_sentiment.csv`
  - `data/processed/scalers/feature_scaler.pkl`
  - `data/processed/scalers/target_scaler.pkl`
- Legacy model artifacts:
  - `models/model_config.json`
  - `models/best_model.pth`
  - `models/backtest_result.png`
  - `models/backtest_weekly_3class.png`
  - `models/backtest_weekly_binary.png`
  - `models/backtest_weekly_comparison.png`
  - `models/best_model_weekly_3class.pth`
  - `models/training_history.png`
  - `models/training_history_weekly_3class.png`
- Old helper docs:
  - `structure.txt`
  - old notebook scripts (unless doing historical research)

## 13) Minimal "where to look first" list for Nell

If he wants control quickly, start with these files only:

1. `/Users/ryanlu/Projects/Nell/fyp/README.md`
2. `/Users/ryanlu/Projects/Nell/fyp/openbb_refresh.py`
3. `/Users/ryanlu/Projects/Nell/fyp/train_weekly.py`
4. `/Users/ryanlu/Projects/Nell/fyp/backtest_weekly.py`
5. `/Users/ryanlu/Projects/Nell/fyp/weekly_inference.py`
6. `/Users/ryanlu/Projects/Nell/fyp/src/config.py`
7. `/Users/ryanlu/Projects/Nell/fyp/src/utils.py`
8. `/Users/ryanlu/Projects/Nell/fyp/src/data/openbb_ingestion.py`
9. `/Users/ryanlu/Projects/Nell/fyp/web/src/lib/data/signal-repository.ts`
10. `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/dashboard/page.tsx`
11. `/Users/ryanlu/Projects/Nell/fyp/web/src/app/(terminal)/openbb/page.tsx`

## 14) Runtime flow summary (current weekly architecture)

1. `openbb_refresh.py` pulls market data and updates:
   - `data/raw/openbb/*.csv`
   - `data/raw/openbb/manifest.json`
   - `models/openbb_refresh_status.json`
   - optionally `data/processed/training_data_openbb.csv`
2. `train_weekly.py` trains and writes weekly model artifacts.
3. `backtest_weekly.py` writes walk-forward metrics and evaluation plot.
4. `weekly_inference.py` emits latest JSON signal and updates signal history.
5. Next.js app calls Python scripts via repository/services layer and renders pages/APIs.

This is the active path that drives the current dashboard.
