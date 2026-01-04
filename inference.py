import torch
import pandas as pd
import numpy as np
import json
import joblib
import sys
import os

# Add src to path to ensure imports work
sys.path.append(os.path.abspath('.'))

from src.config import *
from src.model import TransformerModel
from src.model import LSTMModel
# Note: We load scalers via joblib directly for safety, 
# or you can use your utils.load_scaler if it points to the right path.

# Ensure DEVICE is defined
if 'DEVICE' not in locals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_next_day(model, recent_data, device):
    """Predict the next day's Close Price (Scaled)"""
    model.eval()
    with torch.no_grad():
        # Shape: (1, seq_len, input_dim)
        input_tensor = torch.FloatTensor(recent_data).unsqueeze(0).to(device)
        prediction = model(input_tensor)
    return prediction.cpu().item()

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # ---------------------------------------------------------
    # 1. LOAD CONFIGURATION
    # ---------------------------------------------------------
    try:
        with open(f'{MODEL_PATH}model_config.json', 'r') as f:
            config_data = json.load(f)
        feature_cols = config_data['feature_cols']
        input_dim = config_data['input_dim']
        print(f"Loaded config. Using {input_dim} features.")
    except FileNotFoundError:
        print("Error: model_config.json not found. Run train.py first.")
        exit()

    # ---------------------------------------------------------
    # 2. LOAD SCALERS
    # ---------------------------------------------------------
    try:
        feature_scaler = joblib.load(f'{SCALER_PATH}feature_scaler.pkl')
        target_scaler = joblib.load(f'{SCALER_PATH}target_scaler.pkl')
        print("Scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        exit()

    # ---------------------------------------------------------
    # 3. LOAD & PREPARE DATA
    # ---------------------------------------------------------
    # We load the RAW data (because we fixed data_merging.ipynb to save raw)
    df_raw = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # Get the last SEQ_LEN rows for prediction
    # We take a slightly larger slice to ensure we have enough, then tail it
    df_recent = df_raw.tail(SEQ_LEN).copy()

    if len(df_recent) < SEQ_LEN:
        print(f"Error: Not enough data. Need {SEQ_LEN} rows, got {len(df_recent)}")
        exit()

    # Capture the REAL last close price (before scaling) for reporting
    last_actual_close = df_recent['Close'].iloc[-1]

    # SCALE THE FEATURES
    # IMPORTANT: We use transform(), NOT fit()
    try:
        df_recent[feature_cols] = feature_scaler.transform(df_recent[feature_cols])
    except KeyError as e:
        print(f"Column mismatch error: {e}")
        print("Ensure feature_cols in config match columns in csv.")
        exit()

    # Convert to numpy array for the model
    recent_data = df_recent[feature_cols].values

    # ---------------------------------------------------------
    # 4. LOAD MODEL
    # ---------------------------------------------------------
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=1
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth', map_location=DEVICE))
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # ---------------------------------------------------------
    # 5. PREDICT
    # ---------------------------------------------------------
    # The model outputs a Scaled Price (0-1)
    prediction_scaled = predict_next_day(model, recent_data, DEVICE)
    print(f"Raw Scaled Prediction: {prediction_scaled:.4f}")

    # ---------------------------------------------------------
    # 6. INVERSE TRANSFORM
    # ---------------------------------------------------------
    # Convert scaled price back to dollars
    # scaler expects shape (n_samples, n_features), so we pass [[value]]
    predicted_price = target_scaler.inverse_transform([[prediction_scaled]])[0][0]

    # Calculate metrics
    change_dollar = predicted_price - last_actual_close
    change_pct = (change_dollar / last_actual_close) * 100
    direction = "UP" if change_pct > 0 else "DOWN"

    print("-" * 30)
    print(f"PREDICTION REPORT")
    print("-" * 30)
    print(f"Last Actual Close:    ${last_actual_close:.2f}")
    print(f"Predicted Next Close: ${predicted_price:.2f}")
    print(f"Predicted Move:       {direction} ({change_pct:+.2f}%)")
    print("-" * 30)
