import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
from src.model import TransformerModel, LSTMModel 

# Import project modules
from src.config import *
from src.dataset import StockDataset
from src.model import TransformerModel, DirectionalLoss
from src.utils import plot_training_loss, create_sequences, load_scaler

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = DirectionalLoss(alpha=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Reduced patience to 3 to adapt faster
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

        for sequences, targets in loop:
            sequences = sequences.float().to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.float().to(device)
                targets = targets.float().to(device)

                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), f'{MODEL_PATH}best_model.pth')
            patience_counter = 0
            print(" Model saved (New best validation loss)")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(" Early stopping triggered")
                break

    return train_losses, val_losses

if __name__ == "__main__":
    print("Loading processed data...")
    if not os.path.exists(f'{PROCESSED_DATA_PATH}training_data.csv'):
        raise FileNotFoundError("Run data processing notebooks first!")

    # 1. LOAD RAW DATA
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')

    # 2. DEFINE FEATURES
    # Drop 'Target' and 'Return' from features (prevent leakage)
    drop_cols = ['Date', 'Close', 'Target', 'Return'] 
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # Save Config
    config_data = {"feature_cols": feature_cols, "input_dim": len(feature_cols)}
    with open(f'{MODEL_PATH}model_config.json', 'w') as f:
        json.dump(config_data, f)

    # 3. SPLIT INDICES
    total_len = len(df)
    train_end = int(total_len * TRAIN_SPLIT)
    train_indices = range(0, train_end)

    # 4. SCALING (Use StandardScaler for Features)
    from sklearn.preprocessing import StandardScaler
    
    feature_scaler = StandardScaler()
    feature_scaler.fit(df.iloc[train_indices][feature_cols])
    df[feature_cols] = feature_scaler.transform(df[feature_cols])

    # NOTE: We do NOT scale the Target (Returns) to 0-1. 
    # Returns are already small numbers centered on 0.
    # We just use them directly.
    
    # Save Scaler
    os.makedirs(SCALER_PATH, exist_ok=True)
    joblib.dump(feature_scaler, f'{SCALER_PATH}feature_scaler.pkl')
    
    # 5. CREATE SEQUENCES
    features = df[feature_cols].values
    targets = df['Target'].values # predicting Return
    
    X, y = create_sequences(features, targets, SEQ_LEN)
    
    print(f"Total sequences created: {len(X)}")

    # Split the *sequences* into Train/Val/Test
    # We use the same ratios on the sequence array
    total_seqs = len(X)
    train_seq_end = int(total_seqs * TRAIN_SPLIT)
    val_seq_end = int(total_seqs * (TRAIN_SPLIT + VAL_SPLIT))

    X_train = X[:train_seq_end]
    y_train = y[:train_seq_end]
    
    X_val = X[train_seq_end:val_seq_end]
    y_val = y[train_seq_end:val_seq_end]

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 6. DATA LOADERS
    # ---------------------------------------------------------
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    use_pin_memory = True if DEVICE.type == 'cuda' else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  
        pin_memory=use_pin_memory,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        pin_memory=use_pin_memory,
        num_workers=0
    )

    # ---------------------------------------------------------
    # 7. MODEL SETUP & TRAINING
    # ---------------------------------------------------------
    input_dim = X_train.shape[2]
    
    model = LSTMModel(input_dim=len(feature_cols)).to(DEVICE)
    
    print(f"Model initialized on {DEVICE}")
    
    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, DEVICE
        )

        plt = plot_training_loss(train_losses, val_losses)
        plt.savefig(f'{MODEL_PATH}training_history.png')
        print(f"Training complete! Chart saved to {MODEL_PATH}training_history.png")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
