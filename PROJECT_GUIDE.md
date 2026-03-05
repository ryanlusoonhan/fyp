# Project Guide

## 1) What this project is

- Goal: weekly HSI direction signal (`BUY` / `NO_BUY`) with backtest + web dashboard.
- Model: PyTorch LSTM classifier.
- Main data source for training: `data/processed/training_data.csv` (professor dataset).
- OpenBB role: secondary market context + refresh/status + optional secondary dataset (`training_data_openbb.csv`).

## 2) The 6 files that control almost everything

- `train_weekly.py` - trains model and writes model/scaler/config artifacts.
- `backtest_weekly.py` - evaluates model and writes walk-forward/backtest outputs.
- `weekly_inference.py` - generates latest signal and signal history CSV.
- `scenario_weekly.py` - threshold sweep (scenario lab backend).
- `src/utils.py` - feature engineering + labels + sequence/split logic.
- `web/src/lib/data/signal-repository.ts` - web bridge between Next.js and Python scripts.

## 3) How data flows end-to-end

1. Optional OpenBB refresh  
   `openbb_refresh.py` -> `src/data/openbb_ingestion.py` -> writes `data/raw/openbb/*` and `models/openbb_refresh_status.json`.
2. Training  
   `train_weekly.py` reads training CSV, builds features/labels, trains model.
3. Artifacts generated  
   `models/best_model_weekly_binary.pth`, `models/model_config_weekly.json`, `data/processed/scalers/feature_scaler_weekly.pkl`.
4. Backtest  
   `backtest_weekly.py` writes `models/backtest_weekly_walk_forward.csv` and chart PNG.
5. Inference  
   `weekly_inference.py` outputs latest signal JSON and appends `models/signal_history_weekly.csv`.
6. Web app  
   Next.js pages read those artifacts through repository files in `web/src/lib/data/*`.

## 4) Folder map (what to edit vs ignore)

### Core Python (edit often)

- `src/config.py` - constants, paths, symbol lists.
- `src/utils.py` - features, triple-barrier labels, sequence/split helpers.
- `src/model.py` - LSTM architecture.
- `src/data/openbb_client.py` - OpenBB SDK wrapper/retry/normalization.
- `src/data/openbb_ingestion.py` - OpenBB snapshot building.
- `src/data/hybrid_data.py` - optional hybrid merge/sentiment helpers (not main default path).

### Main scripts (edit often)

- `train_weekly.py`
- `backtest_weekly.py`
- `weekly_inference.py`
- `scenario_weekly.py`
- `openbb_refresh.py`

### Web app (edit for UI/API behavior)

- `web/src/app/(terminal)/*` - dashboard pages.
- `web/src/components/*` - UI components/charts.
- `web/src/lib/data/*` - reads model artifacts, invokes Python.
- `web/src/app/api/*` - route handlers.

### Tests (edit when behavior changes)

- `tests/*` - Python tests.
- `web/src/**/*.spec.ts` - web tests.

### Legacy / historical (usually ignore)

- `train.py`, `backtest.py`, `inference.py`
- `notebooks/*`
- `docs/gtm/*`, `.claude/*` (business/marketing context)

## 5) Web page map

- `/dashboard` - latest signal, KPI cards, equity curve, data refresh status.
- `/signals` - historical signal table.
- `/walk-forward` - robustness windows (accuracy/F1/returns).
- `/scenario-lab` - threshold tuning UI.
- `/explainability` - signal narrative + OpenBB context card.
- `/openbb` - OpenBB snapshot/status + model integration view.

## 6) API map (web)

- `GET /api/public/signal/latest` - latest signal payload.
- `GET /api/public/performance/summary` - walk-forward summary.
- `GET /api/pro/signals/history` - signal history.
- `GET /api/pro/explanations/[signalId]` - explanation payload.
- `POST /api/pro/scenario/run` - threshold scenario run.
- `POST /api/signal/refresh` - refresh + infer latest signal.
- `GET /api/health` - health check.

## 7) Files generated during normal usage

- `models/best_model_weekly_binary.pth`
- `models/model_config_weekly.json`
- `data/processed/scalers/feature_scaler_weekly.pkl`
- `models/backtest_weekly_walk_forward.csv`
- `models/signal_history_weekly.csv`
- `models/openbb_refresh_status.json`
- `data/raw/openbb/*` (OpenBB snapshots)

## 8) Normal command sequence (Docker)

```bash
docker compose up -d --build
docker compose exec -T app python openbb_refresh.py --mode batch --start-date 2015-01-01 --end-date $(date +%F)
docker compose exec -T app python train_weekly.py
docker compose exec -T app python backtest_weekly.py --objective return --threshold 0.50 --test-window 100
docker compose exec -T app python weekly_inference.py --objective accuracy --threshold 0.50 --json
```

## 9) “I want to change X” quick map

- Change features -> `src/utils.py` + retrain.
- Change model params -> `src/config.py` and/or `train_weekly.py`.
- Change threshold behavior -> `backtest_weekly.py`, `weekly_inference.py`, `scenario_weekly.py`.
- Change signal history logic -> `weekly_inference.py` + `web/src/lib/data/signal-repository.ts`.
- Change dashboard layout -> `web/src/app/(terminal)/*` + `web/src/components/*`.
- Change OpenBB ingestion -> `src/data/openbb_client.py` + `src/data/openbb_ingestion.py`.

## 10) Current important reality

- Main training path currently uses primary dataset features (`engineer_features_primary_only`).
- OpenBB is present and working for refresh/status/context; it is not the primary training source by default.
