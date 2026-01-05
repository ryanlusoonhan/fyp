import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
import os
from tqdm import tqdm

# Import project modules
from src.config import *
from src.model import LSTMModel, TransformerModel

def load_model_artifacts(device):
    """Loads the trained model, scaler, and configuration."""
    print("Loading model artifacts...")
    
    # 1. Load Config
    config_path = f'{MODEL_PATH}model_config.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. Run train.py first.")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    feature_cols = config['feature_cols']
    input_dim = config['input_dim']
    
    # 2. Load Scaler
    scaler_path = f'{SCALER_PATH}feature_scaler.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}.")
    scaler = joblib.load(scaler_path)
    
    # 3. Load Model
    # IMPORTANT: Initialize the same model class used in training
    model = LSTMModel(input_dim=input_dim).to(device)
    
    model_path = f'{MODEL_PATH}best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, scaler, feature_cols

def run_backtest(days=90, threshold=0.5):
    """
    Runs a rolling window backtest.
    
    Args:
        days (int): Number of trading days to backtest.
        threshold (float): Probability threshold for buying (default 0.5).
    """
    
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running backtest on {device} for last {days} days...")

    # 1. Load Data
    # We load the raw processed data to get prices
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Load Artifacts
    model, scaler, feature_cols = load_model_artifacts(device)
    
    # 3. Data Preparation
    # We need: lookback (SEQ_LEN) + test period (days) + 1 (for tomorrow's target)
    required_len = SEQ_LEN + days + 1
    
    if len(df) < required_len:
        print(f"Warning: Dataset length ({len(df)}) is shorter than requested backtest ({required_len}). Using full available test set.")
        subset = df.copy()
    else:
        # Take the tail of the dataframe
        subset = df.iloc[-required_len:].reset_index(drop=True)

    # Scale features ONCE using the pre-fitted scaler
    # (We are not fitting here, so no leakage)
    subset_scaled = subset.copy()
    subset_scaled[feature_cols] = scaler.transform(subset[feature_cols])
    
    # Convert features to numpy for fast indexing
    feature_data = subset_scaled[feature_cols].values
    close_prices = subset['Close'].values
    dates = subset['Date'].values

    results = []

    # 4. Rolling Prediction Loop
    # We iterate such that 'i' is the index of "Today".
    # We use window [i-SEQ_LEN+1 : i+1] (inclusive of i) to predict return at i+1
    
    # Start index: We need SEQ_LEN data points ending at 'i'. 
    # So i must be at least SEQ_LEN - 1.
    start_idx = SEQ_LEN - 1
    # End index: We stop at len() - 2, because we need i+1 to exist to calculate return.
    end_idx = len(subset) - 2

    print("Executing rolling predictions...")
    for i in tqdm(range(start_idx, end_idx + 1)):
        
        # --- A. Prepare Input (Window ending at Day i) ---
        # Slicing: [start : end] -> Python excludes end, so we use i+1
        seq_start = i - SEQ_LEN + 1
        seq_end = i + 1
        
        sequence = feature_data[seq_start:seq_end] # Shape: (SEQ_LEN, n_features)
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        
        # --- B. Inference ---
        with torch.no_grad():
            output = model(seq_tensor)
            prob = torch.sigmoid(output).item()
            
        # Strategy Signal
        pred_signal = 1 if prob > threshold else 0
        
        # --- C. Calculate Forward Outcome ---
        # We are at Day i. We trade at Close of Day i (or Open of i+1).
        # We hold until Close of Day i+1.
        price_today = close_prices[i]
        price_tomorrow = close_prices[i+1]
        date_tomorrow = dates[i+1]
        
        # Actual Return for holding 1 day
        actual_return = (price_tomorrow - price_today) / price_today
        
        # Determine strict target (did it go up?)
        true_target = 1 if actual_return > 0 else 0
        
        results.append({
            'Date': date_tomorrow,
            'Price_In': price_today,
            'Price_Out': price_tomorrow,
            'Probability': prob,
            'Signal': pred_signal,
            'Actual_Return': actual_return,
            'Target': true_target
        })

    # 5. Analysis & Metrics
    results_df = pd.DataFrame(results)
    
    # Calculate Strategy Return
    # If Signal=1, we get Actual_Return. If Signal=0, we get 0 (Cash).
    # (Optional: Subtract transaction costs here, e.g., -0.001 for 0.1% fee)
    results_df['Strategy_Return'] = results_df['Signal'] * results_df['Actual_Return']
    
    # Cumulative Returns
    results_df['Cum_Market'] = (1 + results_df['Actual_Return']).cumprod()
    results_df['Cum_Strategy'] = (1 + results_df['Strategy_Return']).cumprod()
    
    # Key Metrics
    total_trades = results_df['Signal'].sum()
    accuracy = (results_df['Signal'] == results_df['Target']).mean()
    
    # Precision: When we bought, were we right?
    buys = results_df[results_df['Signal'] == 1]
    precision = (buys['Actual_Return'] > 0).mean() if len(buys) > 0 else 0
    
    market_ret = results_df['Cum_Market'].iloc[-1] - 1
    strategy_ret = results_df['Cum_Strategy'].iloc[-1] - 1
    
    print("\n" + "="*40)
    print(f"BACKTEST RESULTS ({len(results_df)} Days)")
    print("="*40)
    print(f"Total Trades: {int(total_trades)}")
    print(f"Accuracy (Direction): {accuracy:.2%}")
    print(f"Precision (Win Rate): {precision:.2%}")
    print(f"Market Return: {market_ret:.2%}")
    print(f"Strategy Return: {strategy_ret:.2%}")
    print(f"Outperformance: {strategy_ret - market_ret:.2%}")
    print("="*40)
    
    # Debugging Confidence
    print("\nModel Probability Stats:")
    print(results_df['Probability'].describe())
    
    # 6. Plotting
    plt.figure(figsize=(12, 6))
    
    plt.plot(results_df['Date'], results_df['Cum_Market'], 
             label='Market (Buy & Hold)', color='gray', linestyle='--', alpha=0.6)
    
    plt.plot(results_df['Date'], results_df['Cum_Strategy'], 
             label='Strategy (AI)', color='blue', linewidth=2)
    
    # Highlight winning trades
    wins = results_df[(results_df['Signal'] == 1) & (results_df['Actual_Return'] > 0)]
    plt.scatter(wins['Date'], wins['Cum_Strategy'], marker='^', color='green', alpha=0.6, s=30)
    
    # Highlight losing trades
    losses = results_df[(results_df['Signal'] == 1) & (results_df['Actual_Return'] <= 0)]
    plt.scatter(losses['Date'], losses['Cum_Strategy'], marker='v', color='red', alpha=0.6, s=30)
    
    plt.title(f'Backtest: AI vs Market (Last {len(results_df)} Days)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (1.0 = Breakeven)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f'{MODEL_PATH}backtest_verified.png'
    plt.savefig(save_path)
    print(f"\nChart saved to {save_path}")
    
    # Save CSV for detailed inspection
    results_df.to_csv(f'{MODEL_PATH}backtest_detailed.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    try:
        run_backtest(days=100, threshold=0.5)
    except Exception as e:
        print(f"An error occurred: {e}")
