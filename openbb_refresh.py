import argparse
import json

from src.data.openbb_ingestion import run_openbb_refresh


def parse_args():
    parser = argparse.ArgumentParser(description="Refresh OpenBB snapshots and generate processed training data.")
    parser.add_argument("--mode", choices=["batch", "live"], default="batch", help="Refresh mode.")
    parser.add_argument("--start-date", type=str, default=None, help="Optional refresh start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Optional refresh end date (YYYY-MM-DD).")
    parser.add_argument("--lookback-days", type=int, default=180, help="Lookback days used by live mode.")
    parser.add_argument(
        "--write-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to write data/processed/training_data_openbb.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    status = run_openbb_refresh(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        write_training=args.write_training,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
