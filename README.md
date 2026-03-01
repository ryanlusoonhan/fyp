# Stock Prediction Interface

Simple Docker-first run instructions.

## Prerequisites

- Docker Desktop installed
- Docker daemon running

## Run the app

From project root:

```bash
cp web/.env.example web/.env.local
docker compose up -d --build
```

Open:

- http://localhost:3000 (main dashboard)
- http://localhost:3000/openbb (OpenBB status page)

## Run model commands (inside Docker)

Use a second terminal in project root:

```bash
# 1) Refresh market data from OpenBB
docker compose exec -T app python openbb_refresh.py --mode batch --start-date 2015-01-01 --end-date $(date +%F)

# 2) Train weekly model
docker compose exec -T app python train_weekly.py

# 3) Backtest weekly model
docker compose exec -T app python backtest_weekly.py --objective return

# 4) Generate latest signal (JSON)
docker compose exec -T app python weekly_inference.py --objective return --json
```

Optional quick live refresh:

```bash
docker compose exec -T app python openbb_refresh.py --mode live --lookback-days 30 --no-write-training
```

## Stop everything

```bash
docker compose down
```

## If something fails

```bash
docker compose logs -f app
docker compose down
docker compose up -d --build
```
