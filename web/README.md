# Stock Prediction Interface (Web MVP)

Next.js full-stack product layer for the weekly HSI signal engine.

## What this app does

- Public landing + pricing experience
- Terminal-style dashboard for signal operations
- APIs for:
  - public preview (`/api/public/*`)
  - pro/elite intelligence (`/api/pro/*`)
  - billing checkout + portal (`/api/billing/*`)
  - internal model ingest (`/api/internal/model/ingest`)

## Prerequisites

- Node.js 20+
- Python virtualenv in repo root (`../.venv`) with model dependencies installed
- Weekly artifacts generated in root project:
  - `models/best_model_weekly_binary.pth`
  - `models/model_config_weekly.json`
  - `data/processed/scalers/feature_scaler_weekly.pkl`
  - `models/backtest_weekly_walk_forward.csv`

## Local setup

1) Install web dependencies:

```bash
cd web
npm install
```

2) Create env file:

```bash
cp .env.example .env.local
```

3) Start dev server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment variables

### Core (optional but recommended)

- `NEXT_PUBLIC_APP_URL` - app base URL (for Stripe redirects)
- `PYTHON_BIN` - override Python executable used to run `weekly_inference.py`
- `NELL_REPO_ROOT` - absolute path to the root Python/model repo

### Stripe (for live billing)

- `STRIPE_SECRET_KEY`
- `STRIPE_PRICE_PRO_MONTHLY`
- `STRIPE_PRICE_PRO_ANNUAL`
- `STRIPE_PRICE_ELITE_MONTHLY`
- `STRIPE_PRICE_ELITE_ANNUAL`

If missing, checkout endpoints run in safe stub mode.

### Supabase (for auth + persistence)

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

### Internal ingest auth

- `INTERNAL_INGEST_TOKEN` (required for `/api/internal/model/ingest`)

## Plan simulation in local

Use a request header to simulate entitlements quickly:

- `x-plan-id: free`
- `x-plan-id: pro`
- `x-plan-id: elite`

You can also use bearer dev tokens (`Authorization: Bearer plan:pro`).

## Verification commands

```bash
npm test
npm run lint
npm run typecheck
npm run build
```

## Supabase migration

Schema lives at:

- `supabase/migrations/20260220_000001_mvp_schema.sql`

Apply with Supabase CLI:

```bash
supabase db push
```
