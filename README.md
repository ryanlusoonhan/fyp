# Nell FYP

Hybrid repository with:

1. **Weekly model pipeline (Python)** in repo root
2. **Stock Prediction Interface web app (Next.js)** in `web/`

## Quick start

### 1) Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train/evaluate weekly model

```bash
python train_weekly.py
python backtest_weekly.py --objective return
python weekly_inference.py --objective return
```

### 3) Run web app

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Important files

- `weekly_inference.py` - latest weekly signal inference (supports `--json`)
- `backtest_weekly.py` - weekly backtest + walk-forward export
- `models/backtest_weekly_walk_forward.csv` - walk-forward data consumed by web charts
- `web/src/lib/data/signal-repository.ts` - bridge from Next.js to Python inference
- `web/src/app/(terminal)/*` - terminal dashboard pages

## Product docs

- Architecture: `docs/architecture/2026-02-20-signal-terminal-mvp.md`
- GTM/pricing: `docs/gtm/pricing-and-go-to-market.md`
- Marketing context: `.claude/product-marketing-context.md`
