# Nell Signal Terminal MVP Architecture (2026-02-20)

## Scope

This document covers the full-stack MVP architecture for turning the weekly model workflow into a productized terminal.

## High-Level Topology

1. **Python model layer (repo root)**  
   - `train_weekly.py` produces model artifacts.  
   - `backtest_weekly.py` produces walk-forward metrics (`models/backtest_weekly_walk_forward.csv`).  
   - `weekly_inference.py` emits latest signal and supports machine-readable `--json`.

2. **Next.js product layer (`web/`)**  
   - App Router UI for landing, pricing, and terminal pages (`/dashboard`, `/signals`, `/walk-forward`, `/scenario-lab`, `/explainability`).  
   - API routes for public data, pro data, billing, and internal ingest.

3. **Persistence + billing layer (optional in local, ready in prod)**  
   - Supabase for user profiles/subscriptions/signal snapshots.  
   - Stripe for subscription checkout and billing portal.

## Data Flow

1. Model generates latest weekly prediction.
2. Next server reads signal by invoking Python (`weekly_inference.py --json`).
3. Walk-forward CSV is parsed into chart-friendly structures.
4. API layer gates premium endpoints by plan entitlement.
5. Internal ingest endpoint can persist signal snapshots into Supabase.

## Plan & Entitlement Strategy

- Local/dev: plan can be simulated via `x-plan-id` header or bearer token pattern (`plan:pro`, `demo_elite`).
- Production: bearer token can be resolved through Supabase auth, with plan sourced from `subscriptions`/`profiles`.
- Feature gates are centralized in `web/src/lib/domain/entitlements.ts`.

## Billing Strategy

- Checkout API supports live Stripe sessions when env vars are present:
  - `STRIPE_SECRET_KEY`
  - `STRIPE_PRICE_PRO_MONTHLY`
  - `STRIPE_PRICE_PRO_ANNUAL`
  - `STRIPE_PRICE_ELITE_MONTHLY`
  - `STRIPE_PRICE_ELITE_ANNUAL`
- If absent, checkout safely returns a setup-required response.

## Core Trade-Offs

### A) Python invocation from Next.js vs full service extraction
- **Decision**: keep direct process invocation for MVP.
- **Why**: fastest path from existing model scripts to product.
- **Cost**: less scalable than a dedicated model microservice.
- **Future**: move to queue/worker inference service if traffic rises.

### B) Header-based plan simulation in MVP
- **Decision**: keep plan simulation fallback for local iteration.
- **Why**: allows product/UI development before full auth rollout.
- **Cost**: not sufficient for production security.
- **Future**: enforce Supabase JWT-only plan resolution in production mode.

### C) CSV-backed walk-forward repository
- **Decision**: source walk-forward from generated CSV with fallback dataset.
- **Why**: aligns with current Python pipeline and avoids DB dependency early.
- **Cost**: limited historical querying/aggregation in MVP.
- **Future**: mirror into Supabase `signal_snapshots` + analytic tables.

## Non-Functional Targets (MVP)

- **Latency**: API responses < 1s for cached/static reads; < 5s for Python invocation.
- **Reliability**: graceful fallback signal + fallback walk-forward dataset.
- **Security**: token-protected internal ingest endpoint, isolated Stripe secret handling.
- **Maintainability**: domain logic separated into reusable libraries and tested units.

## Next Milestones

1. Replace simulated plan headers with strict Supabase-authenticated entitlement resolution.
2. Add webhook handling for Stripe subscription lifecycle synchronization.
3. Persist and version walk-forward windows in database for richer analytics.
4. Add alert delivery channels (email/Telegram) behind Pro/Elite entitlements.
