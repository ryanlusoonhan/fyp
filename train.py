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
    create_sequences,
    engineer_features_past_only,
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


def train_model(model, train_loader, val_loader, num_epochs, device, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 15

    print(f"Starting directional-classification training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for sequences, targets in loop:
            sequences = sequences.to(device)
            targets = targets.long().to(device)

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct_train / max(1, total_train))

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_acc = correct_train / max(1, total_train)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.long().to(device)

                logits = model(sequences)
                loss = criterion(logits, targets)
                val_loss += loss.item()

                predicted = torch.argmax(logits, dim=1)
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct_val / max(1, total_val)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={avg_train_loss:.4f} (Acc={train_acc:.2%}) | "
            f"Val Loss={avg_val_loss:.4f} (Acc={val_acc:.2%})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_PATH}best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at Epoch {epoch + 1}")
                break

    return train_losses, val_losses


if __name__ == "__main__":
    set_seed(42)

    path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    print("Loading processed data...")
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    df = engineer_features_past_only(df, sentiment_window=SENTIMENT_WINDOW)

    if "Target" in df.columns:
        df["Target"] = df["Target"].astype(int)
    else:
        if "Close" not in df.columns:
            raise ValueError("'Close' column is required to build directional target.")
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.iloc[:-1].copy()

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions.")

    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=SEQ_LEN)
    if len(df_train) <= SEQ_LEN or len(df_val) <= SEQ_LEN:
        raise ValueError(
            f"Not enough rows after split for seq_len={SEQ_LEN}. "
            f"train={len(df_train)}, val={len(df_val)}"
        )

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    os.makedirs(SCALER_PATH, exist_ok=True)
    joblib.dump(scaler, f"{SCALER_PATH}feature_scaler.pkl")

    train_feat = scaler.transform(df_train[feature_cols].values)
    val_feat = scaler.transform(df_val[feature_cols].values)

    y_train_raw = df_train["Target"].astype(int).values
    y_val_raw = df_val["Target"].astype(int).values

    X_train, y_train = create_sequences(train_feat, y_train_raw, SEQ_LEN)
    X_val, y_val = create_sequences(val_feat, y_val_raw, SEQ_LEN)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Sequence generation produced an empty split. "
            f"X_train={X_train.shape}, X_val={X_val.shape}"
        )

    print(f"Sequences -> X_train: {X_train.shape}, X_val: {X_val.shape}")

    class_weights = compute_class_weights(y_train, num_classes=2)
    print(f"Class weights: {class_weights.tolist()}")

    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f"{MODEL_PATH}model_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "directional_classification",
                "feature_cols": feature_cols,
                "input_dim": len(feature_cols),
                "seq_len": SEQ_LEN,
                "num_classes": 2,
                "label_map": {"0": "DOWN_OR_FLAT", "1": "UP"},
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
        NUM_EPOCHS,
        DEVICE,
        class_weights,
    )

    plt_obj = plot_training_loss(train_losses, val_losses)
    plt_obj.savefig(f"{MODEL_PATH}training_history.png")
    print(f"Training complete. Saved to {MODEL_PATH}best_model.pth")
