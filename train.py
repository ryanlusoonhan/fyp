import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import joblib
import matplotlib.pyplot as plt
import json

# Import project modules
from src.config import *
from src.dataset import StockDataset
from src.model import LSTMModel
from src.utils import plot_training_loss, create_sequences

def train_model(model, train_loader, val_loader, num_epochs, device, pos_weight_value=1.0):
    # --- LOSS FUNCTION ---
    # BCEWithLogitsLoss is best for Binary Classification (0 vs 1)
    # It includes Sigmoid internally for numerical stability
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- OPTIMIZER ---
    # Added weight_decay=1e-5 for L2 Regularization (Prevents Overfitting)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-5 
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15

    print(f"Starting binary classification training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

        for sequences, targets in loop:
            sequences = sequences.float().to(device)
            # Reshape targets to [batch_size, 1] and ensure float for BCELoss
            targets = targets.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            
            outputs = model(sequences) # Shape: [Batch, 1]
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # --- CALCULATE ACCURACY (BINARY) ---
            # Sigmoid > 0.5 means class 1 (Up)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct_train/total_train)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.float().to(device)
                targets = targets.float().view(-1, 1).to(device)

                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} (Acc={train_acc:.2%}) | Val Loss={avg_val_loss:.4f} (Acc={val_acc:.2%})')

        # --- SAVE BEST MODEL & EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{MODEL_PATH}best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch+1}")
                break

    return train_losses, val_losses

if __name__ == "__main__":
    print("Loading processed data...")
    
    if not os.path.exists(f'{PROCESSED_DATA_PATH}training_data.csv'):
        raise FileNotFoundError(f"File not found: {PROCESSED_DATA_PATH}training_data.csv - Run data processing first!")

    # 1. LOAD RAW DATA
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')

    # 2. DEFINE FEATURES
    drop_cols = ['Date', 'Close', 'Target', 'Return', 'Open', 'High', 'Low', 'Volume']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # Save Config
    config_data = {"feature_cols": feature_cols, "input_dim": len(feature_cols)}
    with open(f'{MODEL_PATH}model_config.json', 'w') as f:
        json.dump(config_data, f)

    # 3. SPLIT INDICES
    total_len = len(df)
    train_end = int(total_len * TRAIN_SPLIT)
    train_indices = range(0, train_end)

    # 4. SCALING
    from sklearn.preprocessing import StandardScaler
    feature_scaler = StandardScaler()
    feature_scaler.fit(df.iloc[train_indices][feature_cols])
    df[feature_cols] = feature_scaler.transform(df[feature_cols])

    # Save Scaler
    os.makedirs(SCALER_PATH, exist_ok=True)
    joblib.dump(feature_scaler, f'{SCALER_PATH}feature_scaler.pkl')

    # 5. CREATE SEQUENCES
    features = df[feature_cols].values
    
    # --- CRITICAL FIX: RECALCULATE TARGETS ---
    # We recalculate targets from Close price to ensure we don't get all zeros
    # Shift(-1) compares today's Close with Tomorrow's Close
    # Target = 1 if Tomorrow > Today, else 0
    print("Recalculating targets to ensure validity...")
    df['Recalculated_Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Use fillna(0) for the last row which becomes NaN
    targets = df['Recalculated_Target'].fillna(0).astype(int).values
    
    X, y = create_sequences(features, targets, SEQ_LEN)
    print(f"Total sequences: {len(X)}")

    # Split sequences
    total_seqs = len(X)
    train_seq_end = int(total_seqs * TRAIN_SPLIT)
    
    X_train = X[:train_seq_end]
    y_train = y[:train_seq_end]
    X_val = X[train_seq_end:]
    y_val = y[train_seq_end:]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # --- DEBUGGING DATA DISTRIBUTION ---
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    print(f"DEBUG: Training Data -> Positives (1): {num_pos}, Negatives (0): {num_neg}")
    
    if num_pos == 0 or num_neg == 0:
        print("CRITICAL WARNING: Training data is unbalanced (one class missing). Model will fail.")
        # Fallback to avoid division by zero
        pos_weight_value = 1.0
    else:
        # Weight positive class to handle imbalance
        pos_weight_value = num_neg / num_pos
        print(f"Calculated Positive Weight: {pos_weight_value:.2f}")

    # 6. DATA LOADERS
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    use_pin = True if DEVICE.type == 'cuda' else False
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=use_pin)

    # 7. MODEL SETUP & TRAINING
    # Explicitly using config parameters to prevent overfitting!
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,  # From Config (32)
        num_layers=NUM_LAYERS,  # From Config (1 or 2)
        dropout=DROPOUT,        # From Config (0.5)
        output_dim=1            # Binary Classification
    ).to(DEVICE)

    print(f"Model initialized on {DEVICE} with Hidden={HIDDEN_DIM}, Layers={NUM_LAYERS}, Dropout={DROPOUT}")

    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, DEVICE, pos_weight_value
        )

        plt = plot_training_loss(train_losses, val_losses)
        plt.savefig(f'{MODEL_PATH}training_history.png')
        print(f"Training complete! Chart saved to {MODEL_PATH}training_history.png")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
