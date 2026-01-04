import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.config import *
from src.model import TransformerModel
from src.model import LSTMModel

# Ensure DEVICE is defined
if 'DEVICE' not in locals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def backtest_model(days_to_test=50):
    print(f"Starting Backtest on {DEVICE} for last {days_to_test} days...")

    # 1. Load Config & Raw Data
    try:
        with open(f'{MODEL_PATH}model_config.json', 'r') as f:
            config_data = json.load(f)
        feature_cols = config_data['feature_cols']
        input_dim = config_data['input_dim']
    except FileNotFoundError:
        print("Error: model_config.json not found.")
        return

    # Load RAW data (unscaled)
    df_raw = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # 2. Load Scalers
    try:
        feature_scaler = joblib.load(f'{SCALER_PATH}feature_scaler.pkl')
        target_scaler = joblib.load(f'{SCALER_PATH}target_scaler.pkl')
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return

    # 3. Prepare Data for Backtesting
    # We need a copy of the dataframe to scale features for input
    df_scaled = df_raw.copy()
    
    # Scale features using the training scaler (transform only!)
    df_scaled[feature_cols] = feature_scaler.transform(df_scaled[feature_cols])
    
    # Get arrays
    # Input features (Scaled)
    all_features_scaled = df_scaled[feature_cols].values
    
    # Actual Target Prices (Raw/Unscaled) for comparison
    real_close_prices = df_raw['Close'].values

    # 4. Load Model
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=1
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth', map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")
        return

    model.eval()

    # 5. Backtest Loop
    predictions = []
    actuals = []
    
    # Start index: We need enough history for the first prediction window
    total_len = len(df_raw)
    start_index = total_len - days_to_test
    
    if start_index < SEQ_LEN:
        print(f"Error: Not enough data for {days_to_test} days backtest.")
        return

    print(f"{'Day':<10} {'Actual':<10} {'Predicted':<10} {'Diff %':<10} {'Direction'}")
    print("-" * 55)

    correct_direction_count = 0

    with torch.no_grad():
        for i in range(days_to_test):
            # We want to predict the price at 'current_idx'
            current_idx = start_index + i
            
            # The input window is the SEQ_LEN days BEFORE current_idx
            # Range: [current_idx - SEQ_LEN : current_idx]
            input_window = all_features_scaled[current_idx - SEQ_LEN : current_idx]
            
            # Sanity check
            if len(input_window) != SEQ_LEN:
                continue

            # PREDICT
            input_tensor = torch.FloatTensor(input_window).unsqueeze(0).to(DEVICE)
            
            # Output is Predicted Return (e.g. 0.015 for +1.5%)
            # No inverse_transform needed if we didn't scale the target in train.py
            pred_return = model(input_tensor).cpu().item()
            
            # RECONSTRUCT PRICE
            # Price_Tomorrow = Price_Today * (1 + Pred_Return)
            # We need the UN-SCALED previous close price (Yesterday's close)
            prev_price = real_close_prices[current_idx - 1]
            
            pred_price = prev_price * (1 + pred_return)
            
            # Actual Price (Real $)
            actual_price = real_close_prices[current_idx]
            
            # Metrics
            diff_pct = ((pred_price - actual_price) / actual_price) * 100
            
            # Direction Accuracy
            actual_move = actual_price - prev_price
            pred_move = pred_price - prev_price
            
            # Check if signs match (Up/Up or Down/Down)
            direction_match = (actual_move > 0 and pred_move > 0) or (actual_move < 0 and pred_move < 0)
            
            if direction_match:
                correct_direction_count += 1
                dir_str = "MATCH"
            else:
                dir_str = "WRONG"

            predictions.append(pred_price)
            actuals.append(actual_price)
            
            print(f"{i+1:<10} {actual_price:<10.2f} {pred_price:<10.2f} {diff_pct:<10.2f} {dir_str}")

    # 6. Summary Stats
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    direction_acc = (correct_direction_count / days_to_test) * 100

    print("-" * 55)
    print(f"Backtest Complete.")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional Accuracy: {direction_acc:.2f}%")

    # 7. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Price', color='black')
    plt.plot(predictions, label='Predicted Price', color='blue', linestyle='--')
    plt.title(f'Backtest Last {days_to_test} Days')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{MODEL_PATH}backtest_result.png')
    print(f"Chart saved to {MODEL_PATH}backtest_result.png")

if __name__ == "__main__":
    backtest_model(days_to_test=50)
