import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.config import *
from src.model import LSTMModel
from src.utils import (
    engineer_features_past_only, 
    add_triple_barrier_labels, 
    create_sequences, 
    time_split_with_gap
)

def backtest():
    # 1. LOAD DATA
    path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
        
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
    # 2. PREPROCESS (Must match training exactly)
    df = engineer_features_past_only(df, sentiment_window=SENTIMENT_WINDOW)
    
    # TRIPLE BARRIER LABELING
    # Ensure these match your training settings!
    df = add_triple_barrier_labels(df, barrier_window=10, profit_take=0.03, stop_loss=0.015)
    
    # Split with gap
    df_train, df_val = time_split_with_gap(df, train_split=TRAIN_SPLIT, gap=SEQ_LEN)
    print(f"Original Validation Set: {len(df_val)} rows")
    
    # 3. LOAD SCALER & CONFIG
    scaler_path = f"{SCALER_PATH}feature_scaler_weekly.pkl"
    config_path = f"{MODEL_PATH}model_config_weekly.json"
    
    if not os.path.exists(scaler_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Scaler or Config missing. Train the model first.")
        
    scaler = joblib.load(scaler_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)
        feature_cols = cfg["feature_cols"]
        
    # Transform
    val_feat = scaler.transform(df_val[feature_cols].values)
    y_val_raw = df_val["Target_Class"].astype(int).values
    
    # Create Sequences
    X_val, y_val = create_sequences(val_feat, y_val_raw, SEQ_LEN)
    
    # 4. ALIGN FUTURE RETURNS FOR EQUITY CURVE
    future_ret = df_val["Future_Return"].values
    aligned_future_ret = []
    for i in range(len(val_feat) - SEQ_LEN):
        aligned_future_ret.append(future_ret[i + SEQ_LEN])
    aligned_future_ret = np.array(aligned_future_ret)

    # ==========================================
    # NEW: LIMIT TO LAST 100 WEEKS (TRADING DAYS)
    # ==========================================
    TEST_WINDOW = 100
    if len(X_val) > TEST_WINDOW:
        print(f"Limiting backtest to the last {TEST_WINDOW} periods...")
        X_val = X_val[-TEST_WINDOW:]
        y_val = y_val[-TEST_WINDOW:]
        aligned_future_ret = aligned_future_ret[-TEST_WINDOW:]
    # ==========================================
    
    # 5. LOAD MODEL
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=2, # Binary
    ).to(DEVICE)
    
    model_path = f"{MODEL_PATH}best_model_weekly_binary.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 6. INFERENCE
    probs = []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X_val), batch_size):
            xb = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            logits = model(xb)
            pb = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(pb)
            
    probs = np.vstack(probs)
    
    # --- AGGRESSION TUNING ---
    # Lower threshold to 0.40 to trigger more buys
    pred_class = (probs[:, 1] > 0.40).astype(int) 

    # 7. METRICS
    print("\n" + "="*40)
    print(f"METRICS (Last {len(y_val)} Periods)")
    print("="*40)
    acc = accuracy_score(y_val, pred_class)
    f1 = f1_score(y_val, pred_class, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)
    print(classification_report(y_val, pred_class, digits=4, zero_division=0))
    print("="*40 + "\n")
    
    # 8. TRADING SIMULATION
    initial = 10000.0
    balance = initial
    bh_balance = initial
    equity = [balance]
    bh_equity = [bh_balance]
    
    COST = 0.001 
    step = 1 
    
    for t in range(0, len(pred_class), step):
        action = pred_class[t] # 1=Buy, 0=Wait
        r = aligned_future_ret[t] 
        
        # Buy & Hold
        bh_balance = bh_balance * (1.0 + r)
        
        # AI Strategy
        if action == 1:
            balance = balance * (1.0 - COST) # Entry
            balance = balance * (1.0 + r)    # Hold
            balance = balance * (1.0 - COST) # Exit
        else:
            balance = balance * (1.0 + (r * 0.5)) 
 # Cash
            
        equity.append(balance)
        bh_equity.append(bh_balance)
        
    total_return = (balance - initial) / initial * 100.0
    bh_total_return = (bh_balance - initial) / initial * 100.0
    
    print(f"Final AI Balance: ${balance:.2f} ({total_return:.2f}%)")
    print(f"Final B&H Balance: ${bh_balance:.2f} ({bh_total_return:.2f}%)")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label=f"AI Strategy", color="blue")
    plt.plot(bh_equity, label=f"Buy & Hold", color="gray", linestyle="--", alpha=0.5)
    plt.title(f"Backtest (Last {len(equity)-1} Days) | F1: {f1:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = f"{MODEL_PATH}backtest_limited_100.png"
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    backtest()
