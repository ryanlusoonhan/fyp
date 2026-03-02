# Nell Signal Terminal MVP Architecture (Internal Mode)

## Scope

This document covers the full-stack MVP architecture for turning the weekly model workflow into an internal analytical terminal.

## High-Level Topology

1. **Python model layer (repo root)**  
   - `train_weekly.py` produces model artifacts.  
   - `backtest_weekly.py` produces walk-forward metrics (`models/backtest_weekly_walk_forward.csv`).  
   - `weekly_inference.py` emits latest signal and supports machine-readable `--json`.

2. **Next.js product layer (`web/`)**  
   - App Router UI for landing and terminal pages (`/dashboard`, `/signals`, `/walk-forward`, `/scenario-lab`, `/explainability`, `/openbb`).  
   - API routes for analytics data and internal ingest.

3. **Persistence layer (optional in local, ready in prod)**  
   - Supabase for user profiles/subscriptions/signal snapshots.  

## Data Flow

1. Model generates latest weekly prediction.
2. Next server reads signal by invoking Python (`weekly_inference.py --json`).
3. Walk-forward CSV is parsed into chart-friendly structures.
4. API layer serves all analytical modules in internal mode (no feature-tier gating).
5. Internal ingest endpoint can persist signal snapshots into Supabase.

## Access Strategy

- Internal mode: all analytics endpoints are accessible without pricing-tier checks.
- Optional bearer auth can still be used to resolve a Supabase user id for audit metadata.

## Core Trade-Offs

### A) Python invocation from Next.js vs full service extraction
- **Decision**: keep direct process invocation for MVP.
- **Why**: fastest path from existing model scripts to product.
- **Cost**: less scalable than a dedicated model microservice.
- **Future**: move to queue/worker inference service if traffic rises.

### B) Internal-mode API surface
- **Decision**: remove pricing-tier gating and keep analytics routes universally available.
- **Why**: deployment is for internal academic use, not SaaS monetization.
- **Cost**: no fine-grained feature restrictions.
- **Future**: if needed, add role-based access focused on identity/security rather than pricing.

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

1. Persist and version walk-forward windows in database for richer analytics.
2. Add alert delivery channels (email/Telegram) for internal operations.
3. Add stronger role-based access controls only if required by the supervisor team.
