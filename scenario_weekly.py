import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch

from backtest_weekly import optimize_decision_threshold, run_model_probabilities
from src.config import *
from src.model import LSTMModel
from src.utils import (
    add_triple_barrier_labels,
    create_sequences,
    engineer_features_market_only,
    time_split_with_gap,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run threshold scenario analysis on weekly validation data.")
    parser.add_argument("--objective", choices=["f1", "return"], default="return")
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--cost", type=float, default=0.001)
    parser.add_argument("--barrier-window", type=int, default=None)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def resolve_data_path(config_data_file: str | None, override_data_file: str | None) -> str:
    candidates = [
        override_data_file,
        config_data_file,
        OPENBB_TRAINING_FILE,
        f"{PROCESSED_DATA_PATH}training_data.csv",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No scenario dataset found. Checked: {[c for c in candidates if c]}")


def build_threshold_grid(min_value: float, max_value: float, step: float) -> list[float]:
    low = max(0.01, float(min_value))
    high = min(0.99, float(max_value))
    if low > high:
        low, high = high, low
    stride = max(0.001, float(step))
    values = np.arange(low, high + stride / 2.0, stride)
    return [round(float(value), 4) for value in values]


def run_scenario(args) -> dict:
    config_path = f"{MODEL_PATH}model_config_weekly.json"
    scaler_path = f"{SCALER_PATH}feature_scaler_weekly.pkl"
    model_path = f"{MODEL_PATH}best_model_weekly_binary.pth"

    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config_weekly.json not found. Run train_weekly.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("feature_scaler_weekly.pkl not found. Run train_weekly.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model_weekly_binary.pth not found. Run train_weekly.py first.")

    with open(config_path, "r", encoding="utf-8") as file:
        cfg = json.load(file)

    feature_cols = cfg["feature_cols"]
    seq_len = int(cfg.get("seq_len", SEQ_LEN))
    num_classes = int(cfg.get("num_classes", 2))
    profit_take = float(cfg.get("profit_take", 0.03))
    stop_loss = float(cfg.get("stop_loss", 0.015))
    barrier_window = int(args.barrier_window or cfg.get("barrier_window", 10))

    data_path = resolve_data_path(cfg.get("data_file"), args.data_file)
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    df = engineer_features_market_only(df)
    df = add_triple_barrier_labels(
        df,
        barrier_window=barrier_window,
        profit_take=profit_take,
        stop_loss=stop_loss,
    )

    missing_cols = [column for column in feature_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(
            "Feature columns in weekly config are missing from scenario data: "
            f"{missing_cols}. Re-run train_weekly.py."
        )

    _, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=seq_len)
    if len(df_val) <= seq_len:
        raise ValueError(
            "Not enough validation rows for scenario analysis after split. "
            f"Need > {seq_len}, got {len(df_val)}."
        )

    scaler = joblib.load(scaler_path)
    val_feat = scaler.transform(df_val[feature_cols].values)
    y_val_raw = df_val["Target_Class"].astype(int).values
    X_val, y_val = create_sequences(val_feat, y_val_raw, seq_len)
    if len(X_val) == 0:
        raise ValueError("No validation sequences generated for scenario analysis.")

    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=num_classes,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs = run_model_probabilities(model, X_val, DEVICE)
    probs_up = probs[:, 1]

    future_ret = df_val["Future_Return"].values
    aligned_future_ret = np.array([future_ret[index + seq_len] for index in range(len(val_feat) - seq_len)])
    thresholds = build_threshold_grid(args.threshold_min, args.threshold_max, args.step)

    best_threshold, best_score, table = optimize_decision_threshold(
        probs_up=probs_up,
        y_true=y_val,
        objective=args.objective,
        thresholds=thresholds,
        aligned_future_ret=aligned_future_ret,
        barrier_window=barrier_window,
        cost=args.cost,
    )

    return {
        "objective": args.objective,
        "bestThreshold": float(best_threshold),
        "bestScore": float(best_score),
        "candidates": table.to_dict(orient="records"),
        "nSamples": int(len(y_val)),
        "barrierWindow": barrier_window,
        "dataFile": data_path,
    }


def main():
    args = parse_args()
    result = run_scenario(args)

    if args.json_output:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
