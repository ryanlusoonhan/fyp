import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import *
from src.dataset import StockDataset
from src.model import LSTMModel
from src.utils import (
    add_triple_barrier_labels,
    create_sequences,
    engineer_features_market_only,
    plot_training_loss,
    time_split_with_gap,
)

EXCLUDE_COLS = {
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Target",
    "Return",
    "Recalculated_Target",
    "Target_Class",
    "Future_Close",
    "Future_Return",
}

BARRIER_WINDOW = 10
PROFIT_TAKE = 0.03
STOP_LOSS = 0.015


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def resolve_training_data_path(
    openbb_path: str = OPENBB_TRAINING_FILE,
    fallback_path: str = f"{PROCESSED_DATA_PATH}training_data.csv",
    min_rows: int = max(200, (SEQ_LEN * 6)),
) -> str:
    def has_min_rows(path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as file:
            row_count = sum(1 for _ in file) - 1
        return row_count >= min_rows

    if has_min_rows(openbb_path):
        return openbb_path
    if has_min_rows(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing data file. Checked: {openbb_path}, {fallback_path}")


def train_model(model, train_loader, val_loader, device, class_weights=None, num_epochs=NUM_EPOCHS):
    criterion = (
        nn.CrossEntropyLoss(weight=class_weights.to(device))
        if class_weights is not None
        else nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.long().to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        avg_train = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_losses.append(avg_train)

        model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.long().to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        avg_val = total_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={avg_train:.4f} (Acc={train_acc:.2%}) | "
            f"Val Loss={avg_val:.4f} (Acc={val_acc:.2%})"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_PATH}best_model_weekly_binary.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


if __name__ == "__main__":
    set_seed(42)

    path = resolve_training_data_path()
    data_source = "openbb_yfinance" if os.path.normpath(path) == os.path.normpath(OPENBB_TRAINING_FILE) else "legacy_local"

    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    df = engineer_features_market_only(df)
    df = add_triple_barrier_labels(
        df,
        barrier_window=BARRIER_WINDOW,
        profit_take=PROFIT_TAKE,
        stop_loss=STOP_LOSS,
    )

    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=SEQ_LEN)
    print(f"Rows -> train: {len(df_train)}, val: {len(df_val)}")

    if len(df_train) <= SEQ_LEN or len(df_val) <= SEQ_LEN:
        raise ValueError(
            f"Not enough rows after split for seq_len={SEQ_LEN}. "
            f"train={len(df_train)}, val={len(df_val)}"
        )

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found.")

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    os.makedirs(SCALER_PATH, exist_ok=True)
    joblib.dump(scaler, f"{SCALER_PATH}feature_scaler_weekly.pkl")

    train_feat = scaler.transform(df_train[feature_cols].values)
    val_feat = scaler.transform(df_val[feature_cols].values)

    y_train_raw = df_train["Target_Class"].astype(int).values
    y_val_raw = df_val["Target_Class"].astype(int).values

    X_train, y_train = create_sequences(train_feat, y_train_raw, SEQ_LEN)
    X_val, y_val = create_sequences(val_feat, y_val_raw, SEQ_LEN)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Sequence generation produced an empty split. "
            f"X_train={X_train.shape}, X_val={X_val.shape}"
        )

    print(f"Sequences -> X_train: {X_train.shape}, X_val: {X_val.shape}")

    weights = compute_class_weights(y_train, num_classes=2)
    print(f"Computed Class Weights: {weights}")

    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f"{MODEL_PATH}model_config_weekly.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "directional_classification_weekly",
                "feature_cols": feature_cols,
                "input_dim": len(feature_cols),
                "seq_len": SEQ_LEN,
                "num_classes": 2,
                "label_map": {"0": "NO_BUY", "1": "BUY"},
                "barrier_window": BARRIER_WINDOW,
                "profit_take": PROFIT_TAKE,
                "stop_loss": STOP_LOSS,
                "feature_set_version": "openbb_hsi_v1",
                "data_source": data_source,
                "data_file": path,
                "default_threshold": 0.50,
            },
            f,
        )

    use_pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        StockDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        StockDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=use_pin,
    )

    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=2,
    ).to(DEVICE)

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        class_weights=weights,
    )

    plt_obj = plot_training_loss(train_losses, val_losses)
    plt_obj.savefig(f"{MODEL_PATH}training_history_weekly_binary.png")
    print("Done.")
