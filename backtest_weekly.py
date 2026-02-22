import argparse
import json
import os
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import *
from src.model import LSTMModel
from src.utils import (
    add_triple_barrier_labels,
    create_sequences,
    engineer_features_past_only,
    time_split_with_gap,
)


def simulate_period_strategy(pred_class, aligned_future_ret, barrier_window, initial=10000.0, cost=0.001):
    """
    Simulate strategy on non-overlapping barrier windows.

    `aligned_future_ret[t]` is already a forward return over `barrier_window`, so
    we evaluate only every `barrier_window` steps to avoid overlapping compounding.
    """
    balance = float(initial)
    bh_balance = float(initial)
    equity = [balance]
    bh_equity = [bh_balance]

    step = max(1, int(barrier_window))
    for t in range(0, len(pred_class), step):
        action = int(pred_class[t])
        r = float(aligned_future_ret[t])

        bh_balance = bh_balance * (1.0 + r)

        if action == 1:
            balance = balance * (1.0 - cost)
            balance = balance * (1.0 + r)
            balance = balance * (1.0 - cost)

        equity.append(balance)
        bh_equity.append(bh_balance)

    return equity, bh_equity


def build_walk_forward_slices(n_samples: int, window_size: int, step_size: int):
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive.")
    if n_samples <= 0:
        return []
    if n_samples <= window_size:
        return [(0, n_samples)]

    windows = []
    start = 0
    while start + window_size <= n_samples:
        windows.append((start, start + window_size))
        start += step_size

    if windows and windows[-1][1] < n_samples:
        final_start = max(0, n_samples - window_size)
        final_window = (final_start, n_samples)
        if final_window != windows[-1]:
            windows.append(final_window)

    return windows


def optimize_decision_threshold(
    probs_up: np.ndarray,
    y_true: np.ndarray,
    objective: str = "f1",
    thresholds: Iterable[float] | None = None,
    aligned_future_ret: np.ndarray | None = None,
    barrier_window: int = 10,
    cost: float = 0.001,
):
    objective = objective.lower()
    if objective not in {"f1", "return"}:
        raise ValueError("objective must be one of {'f1', 'return'}.")
    if objective == "return" and aligned_future_ret is None:
        raise ValueError("aligned_future_ret is required for objective='return'.")

    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.01, dtype=np.float64)

    rows = []
    best_threshold = None
    best_score = float("-inf")

    for threshold in thresholds:
        pred_class = (probs_up > float(threshold)).astype(int)
        if objective == "f1":
            score = f1_score(y_true, pred_class, zero_division=0)
        else:
            equity, _ = simulate_period_strategy(
                pred_class=pred_class,
                aligned_future_ret=aligned_future_ret,
                barrier_window=barrier_window,
                initial=1.0,
                cost=cost,
            )
            score = float(equity[-1] - 1.0)

        rows.append(
            {
                "threshold": float(threshold),
                "score": float(score),
                "buy_rate": float(pred_class.mean()) if len(pred_class) else 0.0,
            }
        )
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score, pd.DataFrame(rows)


def run_model_probabilities(model, features: np.ndarray, device, batch_size: int = 256):
    probs = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            xb = torch.tensor(features[i : i + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def summarize_walk_forward(
    y_true: np.ndarray,
    probs_up: np.ndarray,
    aligned_future_ret: np.ndarray,
    threshold: float,
    barrier_window: int,
    window_size: int,
    step_size: int,
    cost: float,
):
    windows = build_walk_forward_slices(
        n_samples=len(y_true),
        window_size=window_size,
        step_size=step_size,
    )
    rows = []
    for idx, (start, end) in enumerate(windows, start=1):
        y_window = y_true[start:end]
        probs_window = probs_up[start:end]
        ret_window = aligned_future_ret[start:end]
        pred_window = (probs_window > threshold).astype(int)

        equity, bh_equity = simulate_period_strategy(
            pred_class=pred_window,
            aligned_future_ret=ret_window,
            barrier_window=barrier_window,
            initial=10000.0,
            cost=cost,
        )
        ai_return_pct = (equity[-1] / 10000.0 - 1.0) * 100.0
        bh_return_pct = (bh_equity[-1] / 10000.0 - 1.0) * 100.0

        rows.append(
            {
                "window_id": idx,
                "start_idx": int(start),
                "end_idx": int(end),
                "n_samples": int(end - start),
                "accuracy": float(accuracy_score(y_window, pred_window)),
                "f1": float(f1_score(y_window, pred_window, zero_division=0)),
                "ai_return_pct": float(ai_return_pct),
                "buy_hold_return_pct": float(bh_return_pct),
            }
        )

    return pd.DataFrame(rows)


def backtest(
    objective: str = "f1",
    threshold: float | None = None,
    test_window: int = 100,
    walk_forward_window: int = 100,
    walk_forward_step: int = 50,
    cost: float = 0.001,
):
    path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    scaler_path = f"{SCALER_PATH}feature_scaler_weekly.pkl"
    config_path = f"{MODEL_PATH}model_config_weekly.json"
    if not os.path.exists(scaler_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Scaler or config missing. Train the weekly model first.")

    scaler = joblib.load(scaler_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    feature_cols = cfg["feature_cols"]
    seq_len = int(cfg.get("seq_len", SEQ_LEN))
    num_classes = int(cfg.get("num_classes", 2))
    barrier_window = int(cfg.get("barrier_window", 10))
    profit_take = float(cfg.get("profit_take", 0.03))
    stop_loss = float(cfg.get("stop_loss", 0.015))

    df = engineer_features_past_only(df, sentiment_window=SENTIMENT_WINDOW)
    df = add_triple_barrier_labels(
        df,
        barrier_window=barrier_window,
        profit_take=profit_take,
        stop_loss=stop_loss,
    )

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Feature columns in weekly config are missing from data: "
            f"{missing_cols}. Re-run train_weekly.py."
        )

    _, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=seq_len)
    print(f"Validation rows: {len(df_val)}")

    val_feat = scaler.transform(df_val[feature_cols].values)
    y_val_raw = df_val["Target_Class"].astype(int).values
    X_val, y_val = create_sequences(val_feat, y_val_raw, seq_len)
    if len(X_val) == 0:
        raise ValueError("No validation sequences were generated for weekly backtest.")

    future_ret = df_val["Future_Return"].values
    aligned_future_ret = np.array([future_ret[i + seq_len] for i in range(len(val_feat) - seq_len)])

    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=num_classes,
    ).to(DEVICE)

    model_path = f"{MODEL_PATH}best_model_weekly_binary.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs = run_model_probabilities(model, X_val, DEVICE)
    probs_up = probs[:, 1]

    if threshold is None:
        threshold, best_score, threshold_table = optimize_decision_threshold(
            probs_up=probs_up,
            y_true=y_val,
            objective=objective,
            aligned_future_ret=aligned_future_ret,
            barrier_window=barrier_window,
            cost=cost,
        )
        score_name = "F1" if objective == "f1" else "Return"
        print(f"Auto-tuned threshold ({objective} objective): {threshold:.2f} | Best {score_name}: {best_score:.4f}")
        print("Top threshold candidates:")
        top = threshold_table.sort_values("score", ascending=False).head(5)
        print(top.to_string(index=False))
    else:
        threshold = float(threshold)
        print(f"Using manual threshold: {threshold:.2f}")

    walk_forward_df = summarize_walk_forward(
        y_true=y_val,
        probs_up=probs_up,
        aligned_future_ret=aligned_future_ret,
        threshold=threshold,
        barrier_window=barrier_window,
        window_size=walk_forward_window,
        step_size=walk_forward_step,
        cost=cost,
    )
    walk_forward_path = f"{MODEL_PATH}backtest_weekly_walk_forward.csv"
    walk_forward_df.to_csv(walk_forward_path, index=False)

    print("\n" + "=" * 40)
    print("WALK-FORWARD SUMMARY")
    print("=" * 40)
    if len(walk_forward_df) > 0:
        means = walk_forward_df[["accuracy", "f1", "ai_return_pct", "buy_hold_return_pct"]].mean()
        print(
            f"Windows: {len(walk_forward_df)} | "
            f"Mean Acc: {means['accuracy']:.4f} | "
            f"Mean F1: {means['f1']:.4f} | "
            f"Mean AI Return: {means['ai_return_pct']:.2f}% | "
            f"Mean B&H Return: {means['buy_hold_return_pct']:.2f}%"
        )
    else:
        print("No walk-forward windows were generated.")
    print(f"Saved walk-forward metrics to: {walk_forward_path}")

    eval_start = 0
    if test_window > 0 and len(y_val) > test_window:
        eval_start = len(y_val) - test_window
        print(f"\nEvaluating latest window: last {test_window} periods...")

    y_eval = y_val[eval_start:]
    probs_eval = probs_up[eval_start:]
    ret_eval = aligned_future_ret[eval_start:]
    pred_class = (probs_eval > threshold).astype(int)

    print("\n" + "=" * 40)
    print(f"WEEKLY METRICS (Evaluation periods: {len(y_eval)})")
    print("=" * 40)
    acc = accuracy_score(y_eval, pred_class)
    f1 = f1_score(y_eval, pred_class, zero_division=0)
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)
    print(classification_report(y_eval, pred_class, digits=4, zero_division=0))

    initial = 10000.0
    equity, bh_equity = simulate_period_strategy(
        pred_class=pred_class,
        aligned_future_ret=ret_eval,
        barrier_window=barrier_window,
        initial=initial,
        cost=cost,
    )

    final_balance = equity[-1]
    final_bh = bh_equity[-1]
    total_return = (final_balance / initial - 1.0) * 100.0
    bh_total_return = (final_bh / initial - 1.0) * 100.0

    print(f"Final AI Balance: ${final_balance:.2f} ({total_return:.2f}%)")
    print(f"Final B&H Balance: ${final_bh:.2f} ({bh_total_return:.2f}%)")

    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="AI Strategy", color="blue")
    plt.plot(bh_equity, label="Buy & Hold", color="gray", linestyle="--", alpha=0.7)
    plt.title(f"Weekly Backtest ({len(equity) - 1} Trades) | F1: {f1:.2f} | Thresh: {threshold:.2f}")
    plt.xlabel("Trade Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = f"{MODEL_PATH}backtest_limited_{len(y_eval)}.png"
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Weekly backtest with threshold tuning and walk-forward evaluation.")
    parser.add_argument(
        "--objective",
        choices=["f1", "return"],
        default="f1",
        help="Objective for automatic threshold tuning.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold. If omitted, threshold is auto-tuned.",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=100,
        help="Evaluate final metrics on the latest N periods (0 means full validation set).",
    )
    parser.add_argument(
        "--walk-forward-window",
        type=int,
        default=100,
        help="Window size for walk-forward evaluation.",
    )
    parser.add_argument(
        "--walk-forward-step",
        type=int,
        default=50,
        help="Step size for walk-forward evaluation.",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.001,
        help="Per-side transaction cost applied in strategy simulation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backtest(
        objective=args.objective,
        threshold=args.threshold,
        test_window=args.test_window,
        walk_forward_window=args.walk_forward_window,
        walk_forward_step=args.walk_forward_step,
        cost=args.cost,
    )
