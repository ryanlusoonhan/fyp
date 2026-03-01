.PHONY: docker-build docker-up docker-down docker-shell docker-refresh docker-train docker-backtest docker-signal docker-pytests docker-webtests

docker-build:
	docker compose build app

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-shell:
	docker compose exec app bash

docker-refresh:
	docker compose exec app python openbb_refresh.py --mode batch --start-date 2015-01-01 --end-date $$(date +%F)

docker-train:
	docker compose exec app python train_weekly.py

docker-backtest:
	docker compose exec app python backtest_weekly.py --objective return

docker-signal:
	docker compose exec app python weekly_inference.py --objective return --json

docker-pytests:
	docker compose exec app python -m unittest discover -s tests -v

docker-webtests:
	docker compose exec app npm --prefix web test
