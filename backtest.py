import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import *
from src.model import LSTMModel
from src.utils import create_sequences
import os

def backtest():
    # 1. SETUP
    print("Loading data for backtest...")
    df = pd.read_csv(f'{PROCESSED_DATA_PATH}training_data.csv')
    
    # Re-create features exactly like train.py
    drop_cols = ['Date', 'Close', 'Target', 'Return', 'Open', 'High', 'Low', 'Volume']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # 2. SCALING (Must load the exact scaler used in training)
    import joblib
    scaler = joblib.load(f'{SCALER_PATH}feature_scaler.pkl')
    
    # Scale ALL data (we will slice out validation later)
    # Note: We suppress the warning because we know we are applying it globally
    # but the scaler was only FIT on train data.
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    features = df_scaled[feature_cols].values
    
    # We need the ACTUAL returns to calculate PnL (Profit and Loss)
    # Recalculate Return shifted to match the Target logic
    # If Target[i] is High, it means Return[i] was positive.
    # We need the Return of the day we are predicting.
    
    # Calculate Next Day Return (Shift -1)
    # This aligns with the target: Target is result of Day T -> T+1
    next_day_returns = df['Close'].pct_change().shift(-1).fillna(0).values
    
    # 3. CREATE SEQUENCES
    # We pass dummy targets because we don't need them for X generation
    dummy_targets = np.zeros(len(features))
    X, _ = create_sequences(features, dummy_targets, SEQ_LEN)
    
    # 4. ALIGN RETURNS WITH SEQUENCES
    # If X[i] is days 0..29, the prediction is for the return at index 30?
    # In train.py: y.append(targets[i + seq_len])
    # So we need returns[i + seq_len]
    aligned_returns = []
    for i in range(len(features) - SEQ_LEN):
        aligned_returns.append(next_day_returns[i + SEQ_LEN])
    aligned_returns = np.array(aligned_returns)
    
    # 5. SPLIT TO VALIDATION ONLY
    total_seqs = len(X)
    train_seq_end = int(total_seqs * TRAIN_SPLIT)
    
    X_val = X[train_seq_end:]
    returns_val = aligned_returns[train_seq_end:]
    
    print(f"Backtesting on {len(X_val)} days of unseen data...")
    
    # 6. LOAD MODEL
    model = LSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=1
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(f'{MODEL_PATH}best_model.pth'))
    model.eval()
    
    # 7. RUN PREDICTION
    predictions = []
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 256
        for i in range(0, len(X_val), batch_size):
            batch_X = torch.tensor(X_val[i:i+batch_size]).float().to(DEVICE)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)
            
    predictions = np.array(predictions)
    
    # 8. SIMULATE TRADING
    initial_balance = 10000
    balance = initial_balance
    equity_curve = [initial_balance]
    
    # Strategy Parameters
    BUY_THRESHOLD = 0.55  # Only buy if model is 55% confident
    # Transaction cost (e.g., 0.1% per trade)
    COST = 0.001 
    
    holdings = 0 # 0 = Cash, 1 = Invested
    
    print("\n--- SIMULATION RESULTS ---")
    
    for i in range(len(predictions)):
        prob = predictions[i]
        actual_ret = returns_val[i]
        
        # LOGIC:
        # If Prob > Threshold, we go LONG (Buy) for the next day
        # If Prob < 0.5, we stay in CASH (Sell/Hold Cash)
        
        if prob > BUY_THRESHOLD:
            # We are invested for this day
            # PnL = Balance * Return - Transaction Cost
            # We apply cost only if we weren't already holding (simplified)
            
            trade_cost = 0
            if holdings == 0:
                trade_cost = COST # Cost to enter
                holdings = 1
                
            # Update balance
            profit = balance * actual_ret
            balance = balance + profit - (balance * trade_cost)
            
        else:
            # We are in cash
            if holdings == 1:
                # Cost to exit
                balance = balance - (balance * COST)
                holdings = 0
                
        equity_curve.append(balance)

    # 9. PLOT
    plt.figure(figsize=(12, 6))
    
    # Create Buy & Hold Benchmark for comparison
    # Cumulative sum of log returns is safest, but for simple comparison:
    # We just track the asset price movement over the same period
    benchmark_returns = returns_val
    benchmark_curve = [initial_balance]
    ben_bal = initial_balance
    for r in benchmark_returns:
        ben_bal = ben_bal * (1 + r)
        benchmark_curve.append(ben_bal)
        
    plt.plot(equity_curve, label='AI Model Strategy', color='green')
    plt.plot(benchmark_curve, label='Buy & Hold Benchmark', color='gray', linestyle='--')
    
    plt.title(f'Backtest: AI vs Buy & Hold\nInitial: ${initial_balance} -> Final: ${balance:.2f}')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{MODEL_PATH}backtest_result.png')
    
    # Stats
    total_return = (balance - initial_balance) / initial_balance * 100
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Chart saved to {MODEL_PATH}backtest_result.png")

if __name__ == "__main__":
    backtest()
