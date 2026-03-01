import argparse
import csv
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from backtest_weekly import optimize_decision_threshold, run_model_probabilities
from src.config import *
from src.data.openbb_ingestion import load_refresh_status, run_openbb_refresh
from src.model import LSTMModel
from src.utils import (
    add_triple_barrier_labels,
    create_sequences,
    engineer_features_market_only,
    time_split_with_gap,
)


def classify_weekly_signal(prob_buy: float, threshold: float):
    pred_class = 1 if float(prob_buy) > float(threshold) else 0
    label = "BUY" if pred_class == 1 else "NO_BUY"
    return pred_class, label


def resolve_inference_data_path(
    config_data_file: str | None = None,
    required_columns: list[str] | None = None,
) -> str:
    required = set(required_columns or [])

    def has_min_rows(path_candidate: str | None, min_rows: int = (SEQ_LEN + 1)) -> bool:
        if not path_candidate or not os.path.exists(path_candidate):
            return False
        with open(path_candidate, "r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1
        return row_count >= min_rows

    def supports_required_features(path_candidate: str) -> bool:
        if not required:
            return True
        try:
            sample_df = pd.read_csv(path_candidate)
            engineered = engineer_features_market_only(sample_df)
        except Exception:
            return False
        return required.issubset(set(engineered.columns))

    candidates = [
        config_data_file,
        OPENBB_TRAINING_FILE,
        f"{PROCESSED_DATA_PATH}training_data.csv",
    ]
    for candidate in candidates:
        if has_min_rows(candidate):
            if required and not supports_required_features(candidate):
                continue
            return candidate
    raise FileNotFoundError(
        "No dataset file found for inference. "
        f"Checked: {[c for c in candidates if c]}"
    )


def build_signal_payload(
    as_of_date: str | None,
    last_close: float | None,
    threshold: float,
    objective: str,
    pred_class: int,
    label: str,
    probability_buy: float,
    model_version: str,
    data_status: str | None = None,
    last_refresh_at: str | None = None,
    latest_market_date: str | None = None,
    data_provider: str | None = None,
) -> dict:
    payload = {
        "as_of_date": as_of_date,
        "last_close": last_close,
        "threshold": float(threshold),
        "objective": objective,
        "pred_class": int(pred_class),
        "label": label,
        "probability_buy": float(probability_buy),
        "probability_no_buy": float(1.0 - probability_buy),
        "model_version": model_version,
    }
    if data_status is not None:
        payload["data_status"] = data_status
    if last_refresh_at is not None:
        payload["last_refresh_at"] = last_refresh_at
    if latest_market_date is not None:
        payload["latest_market_date"] = latest_market_date
    if data_provider is not None:
        payload["data_provider"] = data_provider
    return payload


def append_signal_history(payload: dict, history_path: str = SIGNAL_HISTORY_FILE) -> None:
    row = {
        "as_of_date": payload.get("as_of_date"),
        "last_close": payload.get("last_close"),
        "threshold": payload.get("threshold"),
        "objective": payload.get("objective"),
        "pred_class": payload.get("pred_class"),
        "label": payload.get("label"),
        "probability_buy": payload.get("probability_buy"),
        "probability_no_buy": payload.get("probability_no_buy"),
        "model_version": payload.get("model_version"),
        "data_status": payload.get("data_status"),
        "last_refresh_at": payload.get("last_refresh_at"),
        "latest_market_date": payload.get("latest_market_date"),
    }
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    file_exists = os.path.exists(history_path)
    with open(history_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
    if len(df_val) <= seq_len:
        raise ValueError(
            "Not enough validation rows to infer threshold. "
            f"Need > {seq_len}, got {len(df_val)}."
        )

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
    parser.add_argument(
        "--refresh-openbb",
        action="store_true",
        help="Refresh OpenBB data before inference.",
    )
    parser.add_argument(
        "--refresh-mode",
        choices=["batch", "live"],
        default="live",
        help="Refresh mode when --refresh-openbb is enabled.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=180,
        help="Live refresh lookback window in days.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.json_output:
        print(f"Using device: {DEVICE}")

    config_path = f"{MODEL_PATH}model_config_weekly.json"
    scaler_path = f"{SCALER_PATH}feature_scaler_weekly.pkl"
    model_path = f"{MODEL_PATH}best_model_weekly_binary.pth"
    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config_weekly.json not found. Run train_weekly.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("feature_scaler_weekly.pkl not found. Run train_weekly.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model_weekly_binary.pth not found. Run train_weekly.py first.")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    feature_cols = cfg["feature_cols"]
    input_dim = int(cfg["input_dim"])
    seq_len = int(cfg.get("seq_len", SEQ_LEN))
    num_classes = int(cfg.get("num_classes", 2))
    barrier_window = int(cfg.get("barrier_window", 10))
    profit_take = float(cfg.get("profit_take", 0.03))
    stop_loss = float(cfg.get("stop_loss", 0.015))

    if args.refresh_openbb:
        run_openbb_refresh(
            mode=args.refresh_mode,
            lookback_days=args.lookback_days,
            write_training=False,
        )

    data_path = resolve_inference_data_path(cfg.get("data_file"), required_columns=feature_cols)
    raw_df = pd.read_csv(data_path)
    if "Date" in raw_df.columns:
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        raw_df = raw_df.sort_values("Date").reset_index(drop=True)

    feature_df = engineer_features_market_only(raw_df)
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
        default_threshold = float(cfg.get("default_threshold", 0.50))
        try:
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
        except Exception as error:
            threshold = default_threshold
            if not args.json_output:
                print(
                    "Unable to infer threshold from validation; "
                    f"falling back to default threshold {threshold:.2f}. "
                    f"Reason: {error}"
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
    status = load_refresh_status()
    payload = build_signal_payload(
        as_of_date=as_of_date,
        last_close=last_close,
        threshold=threshold,
        objective=args.objective,
        pred_class=pred_class,
        label=label,
        probability_buy=prob_buy,
        model_version=model_version,
        data_status=status.get("data_status"),
        last_refresh_at=status.get("last_refresh_at"),
        latest_market_date=status.get("latest_market_date"),
        data_provider=status.get("provider"),
    )
    append_signal_history(payload)

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
        if payload.get("data_status") is not None:
            print(f"Data Status:         {payload['data_status']}")
        if payload.get("latest_market_date") is not None:
            print(f"Latest Market Date:  {payload['latest_market_date']}")
        if payload.get("last_refresh_at") is not None:
            print(f"Last Refresh At:     {payload['last_refresh_at']}")
        print("-" * 40)
