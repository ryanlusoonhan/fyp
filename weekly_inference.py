import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from backtest_weekly import optimize_decision_threshold, run_model_probabilities
from src.config import *
from src.model import LSTMModel
from src.utils import (
    add_triple_barrier_labels,
    create_sequences,
    engineer_features_past_only,
    time_split_with_gap,
)


def classify_weekly_signal(prob_buy: float, threshold: float):
    pred_class = 1 if float(prob_buy) > float(threshold) else 0
    label = "BUY" if pred_class == 1 else "NO_BUY"
    return pred_class, label


def infer_threshold_from_validation(
    model,
    feature_df: pd.DataFrame,
    scaler,
    feature_cols: list[str],
    seq_len: int,
    barrier_window: int,
    profit_take: float,
    stop_loss: float,
    objective: str,
    cost: float,
):
    labeled_df = add_triple_barrier_labels(
        feature_df,
        barrier_window=barrier_window,
        profit_take=profit_take,
        stop_loss=stop_loss,
    )
    _, df_val = time_split_with_gap(labeled_df, train_split=TRAIN_SPLIT, gap=seq_len)

    val_feat = scaler.transform(df_val[feature_cols].values)
    y_val_raw = df_val["Target_Class"].astype(int).values
    X_val, y_val = create_sequences(val_feat, y_val_raw, seq_len)
    if len(X_val) == 0:
        raise ValueError("Validation sequence generation is empty while inferring threshold.")

    future_ret = df_val["Future_Return"].values
    aligned_future_ret = np.array([future_ret[i + seq_len] for i in range(len(val_feat) - seq_len)])
    probs = run_model_probabilities(model, X_val, DEVICE)
    probs_up = probs[:, 1]

    return optimize_decision_threshold(
        probs_up=probs_up,
        y_true=y_val,
        objective=objective,
        aligned_future_ret=aligned_future_ret,
        barrier_window=barrier_window,
        cost=cost,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Latest weekly BUY/NO_BUY signal inference.")
    parser.add_argument(
        "--objective",
        choices=["f1", "return"],
        default="f1",
        help="Objective for automatic threshold tuning when --threshold is omitted.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual BUY threshold. If omitted, inferred from validation data.",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.001,
        help="Transaction cost used when tuning threshold with return objective.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit machine-readable JSON output only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.json_output:
        print(f"Using device: {DEVICE}")

    config_path = f"{MODEL_PATH}model_config_weekly.json"
    scaler_path = f"{SCALER_PATH}feature_scaler_weekly.pkl"
    model_path = f"{MODEL_PATH}best_model_weekly_binary.pth"
    data_path = f"{PROCESSED_DATA_PATH}training_data.csv"

    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config_weekly.json not found. Run train_weekly.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("feature_scaler_weekly.pkl not found. Run train_weekly.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model_weekly_binary.pth not found. Run train_weekly.py first.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    feature_cols = cfg["feature_cols"]
    input_dim = int(cfg["input_dim"])
    seq_len = int(cfg.get("seq_len", SEQ_LEN))
    num_classes = int(cfg.get("num_classes", 2))
    barrier_window = int(cfg.get("barrier_window", 10))
    profit_take = float(cfg.get("profit_take", 0.03))
    stop_loss = float(cfg.get("stop_loss", 0.015))

    raw_df = pd.read_csv(data_path)
    if "Date" in raw_df.columns:
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.sort_values("Date").reset_index(drop=True)

    feature_df = engineer_features_past_only(raw_df, sentiment_window=SENTIMENT_WINDOW)
    missing_cols = [c for c in feature_cols if c not in feature_df.columns]
    if missing_cols:
        raise ValueError(
            "Feature columns in weekly config are missing from processed data: "
            f"{missing_cols}. Re-run train_weekly.py."
        )

    scaler = joblib.load(scaler_path)
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=num_classes,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    if args.threshold is None:
        threshold, best_score, _ = infer_threshold_from_validation(
            model=model,
            feature_df=feature_df,
            scaler=scaler,
            feature_cols=feature_cols,
            seq_len=seq_len,
            barrier_window=barrier_window,
            profit_take=profit_take,
            stop_loss=stop_loss,
            objective=args.objective,
            cost=args.cost,
        )
        metric_name = "F1" if args.objective == "f1" else "Return"
        if not args.json_output:
            print(
                f"Auto-tuned threshold ({args.objective} objective): "
                f"{threshold:.2f} | Best {metric_name}: {best_score:.4f}"
            )
    else:
        threshold = float(args.threshold)
        if not args.json_output:
            print(f"Using manual threshold: {threshold:.2f}")

    recent = feature_df.tail(seq_len).copy()
    if len(recent) < seq_len:
        raise ValueError(f"Not enough rows for inference. Need {seq_len}, got {len(recent)}.")

    recent_scaled = scaler.transform(recent[feature_cols].values)
    with torch.no_grad():
        xb = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    prob_buy = float(probs[1])
    pred_class, label = classify_weekly_signal(prob_buy=prob_buy, threshold=threshold)
    prob_no_buy = 1.0 - prob_buy

    last_date = recent["Date"].iloc[-1] if "Date" in recent.columns else None
    last_close = float(recent["Close"].iloc[-1]) if "Close" in recent.columns else None

    as_of_date = pd.Timestamp(last_date).date().isoformat() if last_date is not None else None
    model_version = os.path.basename(model_path)
    payload = {
        "as_of_date": as_of_date,
        "last_close": last_close,
        "threshold": float(threshold),
        "objective": args.objective,
        "pred_class": int(pred_class),
        "label": label,
        "probability_buy": prob_buy,
        "probability_no_buy": prob_no_buy,
        "model_version": model_version,
    }

    if args.json_output:
        print(json.dumps(payload))
    else:
        print("-" * 40)
        print("WEEKLY SIGNAL REPORT")
        print("-" * 40)
        if as_of_date is not None:
            print(f"As-of Date:          {as_of_date}")
        if last_close is not None:
            print(f"Last Close:          ${last_close:.2f}")
        print(f"Decision Threshold:  {threshold:.2f}")
        print(f"Predicted Class:     {pred_class} ({label})")
        print(f"Probability(BUY):    {prob_buy:.2%}")
        print(f"Probability(NO_BUY): {prob_no_buy:.2%}")
        print("-" * 40)
