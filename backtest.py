import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import *
from src.model import LSTMModel
from src.utils import create_sequences, time_split_with_gap


def simulate_daily_strategy(pred_class, aligned_returns, initial=10000.0, cost=0.001):
    balance = float(initial)
    bh_balance = float(initial)
    equity = [balance]
    bh_equity = [bh_balance]

    for action, r in zip(pred_class, aligned_returns):
        bh_balance = bh_balance * (1.0 + r)

        if action == 1:
            balance = balance * (1.0 - cost)
            balance = balance * (1.0 + r)
            balance = balance * (1.0 - cost)

        equity.append(balance)
        bh_equity.append(bh_balance)

    return equity, bh_equity


def backtest():
    print("Loading data for daily backtest...")
    data_path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    if "Target" not in df.columns:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.iloc[:-1].copy()
    else:
        df["Target"] = df["Target"].astype(int)

    scaler_path = f"{SCALER_PATH}feature_scaler.pkl"
    config_path = f"{MODEL_PATH}model_config.json"
    if not os.path.exists(scaler_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Scaler or config missing. Run train.py first.")

    scaler = joblib.load(scaler_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    feature_cols = cfg["feature_cols"]
    seq_len = int(cfg.get("seq_len", SEQ_LEN))
    num_classes = int(cfg.get("num_classes", 2))

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Feature columns in config are missing from training_data.csv: "
            f"{missing_cols}. Re-run train.py."
        )

    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=seq_len)
    print(f"Validation rows: {len(df_val)}")

    val_feat = scaler.transform(df_val[feature_cols].values)
    y_val_raw = df_val["Target"].astype(int).values

    X_val, y_val = create_sequences(val_feat, y_val_raw, seq_len)

    next_day_returns = df_val["Close"].pct_change().shift(-1).fillna(0.0).values
    aligned_returns = np.array(
        [next_day_returns[i + seq_len] for i in range(len(val_feat) - seq_len)]
    )

    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=num_classes,
    ).to(DEVICE)

    model_path = f"{MODEL_PATH}best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X_val), batch_size):
            xb = torch.tensor(X_val[i : i + batch_size], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs)

    buy_threshold = 0.55
    pred_class = (probs[:, 1] > buy_threshold).astype(int)

    print("\n" + "=" * 40)
    print(f"DAILY METRICS (Validation: {len(y_val)} periods)")
    print("=" * 40)
    acc = accuracy_score(y_val, pred_class)
    f1 = f1_score(y_val, pred_class, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)
    print(classification_report(y_val, pred_class, digits=4, zero_division=0))

    equity, bh_equity = simulate_daily_strategy(
        pred_class=pred_class,
        aligned_returns=aligned_returns,
        initial=10000.0,
        cost=0.001,
    )

    final_balance = equity[-1]
    final_bh = bh_equity[-1]
    total_return = (final_balance / 10000.0 - 1.0) * 100.0
    bh_total_return = (final_bh / 10000.0 - 1.0) * 100.0

    print(f"Final AI Balance: ${final_balance:.2f} ({total_return:.2f}%)")
    print(f"Final B&H Balance: ${final_bh:.2f} ({bh_total_return:.2f}%)")

    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="AI Model Strategy", color="green")
    plt.plot(bh_equity, label="Buy & Hold", color="gray", linestyle="--")
    plt.title(f"Daily Directional Backtest | F1: {f1:.2f}")
    plt.xlabel("Periods")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = f"{MODEL_PATH}backtest_result.png"
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")


if __name__ == "__main__":
    backtest()
