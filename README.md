# Nell FYP

Hybrid repository with:

1. **Weekly model pipeline (Python)** in repo root
2. **Stock Prediction Interface web app (Next.js)** in `web/`

## Quick start

### Fastest setup (Docker, recommended)

1) Install Docker Desktop on macOS.

2) From project root, copy web env once:

```bash
cp web/.env.example web/.env.local
```

3) Start the full stack:

```bash
docker compose up --build
```

4) Open [http://localhost:3000](http://localhost:3000).

5) In a second terminal, run model commands inside the same container:

```bash
docker compose exec app python openbb_refresh.py --mode batch --start-date 2015-01-01 --end-date $(date +%F)
docker compose exec app python train_weekly.py
docker compose exec app python backtest_weekly.py --objective return
docker compose exec app python weekly_inference.py --objective return --json
```

Optional shortcuts (same commands via Makefile):

```bash
make docker-refresh
make docker-train
make docker-backtest
make docker-signal
```

Stop containers:

```bash
docker compose down
```

### 1) Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train/evaluate weekly model

```bash
python openbb_refresh.py --mode batch --start-date 2015-01-01 --end-date 2026-03-01
python train_weekly.py
python backtest_weekly.py --objective return
python weekly_inference.py --objective return
```

Live refresh before inference:

```bash
python weekly_inference.py --objective return --json --refresh-openbb --refresh-mode live
```

Manual live snapshot refresh without replacing training dataset:

```bash
python openbb_refresh.py --mode live --lookback-days 180 --no-write-training
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
- `openbb_refresh.py` - OpenBB snapshot refresh + processed dataset generation
- `backtest_weekly.py` - weekly backtest + walk-forward export
- `models/backtest_weekly_walk_forward.csv` - walk-forward data consumed by web charts
- `models/openbb_refresh_status.json` - latest OpenBB refresh diagnostics
- `models/signal_history_weekly.csv` - persisted inference history consumed by `/signals`
- `web/src/lib/data/signal-repository.ts` - bridge from Next.js to Python inference
- `web/src/app/(terminal)/*` - terminal dashboard pages

## Product docs

- Architecture: `docs/architecture/2026-02-20-signal-terminal-mvp.md`
- GTM/pricing: `docs/gtm/pricing-and-go-to-market.md`
- Marketing context: `.claude/product-marketing-context.md`
