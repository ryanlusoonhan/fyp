import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import *
from src.dataset import StockDataset
from src.model import LSTMModel
# IMPORT THE NEW FUNCTION HERE
from src.utils import (
    engineer_features_past_only,
    add_triple_barrier_labels, 
    create_sequences,
    time_split_with_gap,
    plot_training_loss
)

EXCLUDE_COLS = {
    "Date", 
    "Open", "High", "Low", "Close", "Volume",
    "Target", "Return", "Recalculated_Target",
    "Target_Class", "Future_Close", "Future_Return",
}

def compute_class_weights(y: np.ndarray, num_classes: int = 2):
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    # Inverse frequency weights
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

def train_model(model, train_loader, val_loader, device, class_weights=None, num_epochs=NUM_EPOCHS):
    # USE WEIGHTS IF PROVIDED
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience, patience_counter = 15, 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(Xb) # [B, 2]
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
        
        # Validation
        model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                
        avg_val = total_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)
        val_losses.append(avg_val)
        
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} (Acc={train_acc:.2%}) | Val Loss={avg_val:.4f} (Acc={val_acc:.2%})")
        
        # Early Stopping & Saving
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(MODEL_PATH, exist_ok=True)
            # Save as binary model
            torch.save(model.state_dict(), f"{MODEL_PATH}best_model_weekly_binary.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    return train_losses, val_losses

if __name__ == "__main__":
    path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
        
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
    # 1. Feature Engineering
    df = engineer_features_past_only(df, sentiment_window=SENTIMENT_WINDOW)
    
    # 2. LABELING: TRIPLE BARRIER (Profit=3%, Stop=1.5%, Window=10 days)
    # This creates a "Target_Class" of 0 or 1
    df = add_triple_barrier_labels(df, barrier_window=10, profit_take=0.03, stop_loss=0.015)
    
    # 3. Split
    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=SEQ_LEN)
    print(f"Rows -> train: {len(df_train)}, val: {len(df_val)}")
    
    # 4. Scale
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
    
    # 5. Sequences
    X_train, y_train = create_sequences(train_feat, y_train_raw, SEQ_LEN)
    X_val, y_val = create_sequences(val_feat, y_val_raw, SEQ_LEN)
    
    print(f"Sequences -> X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # 6. Class Weights (Crucial for Imbalance)
    weights = compute_class_weights(y_train, num_classes=2)
    print(f"Computed Class Weights: {weights}")
    
    # Save config
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f"{MODEL_PATH}model_config_weekly.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "input_dim": len(feature_cols), "seq_len": SEQ_LEN}, f)
        
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # 7. Train
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=2 # Binary
    ).to(DEVICE)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, DEVICE, class_weights=weights)
    
    plt = plot_training_loss(train_losses, val_losses)
    plt.savefig(f"{MODEL_PATH}training_history_weekly_binary.png")
    print("Done.")
