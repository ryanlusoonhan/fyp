import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from src.config import SCALER_PATH

def engineer_features(df):
    df = df.copy()
    
    # 1. Feature: Smoothed Sentiment (Crucial for noise reduction)
    # News doesn't just vanish; it lingers. We weight the last 7 days.
    if 'sentiment_score' in df.columns:
        df['Sentiment_MA7'] = df['sentiment_score'].rolling(window=7).mean()
        df['Sentiment_Momentum'] = df['sentiment_score'] - df['Sentiment_MA7']
        # Fill NaNs with neutral (0)
        df['Sentiment_MA7'] = df['Sentiment_MA7'].fillna(0)
        df['Sentiment_Momentum'] = df['Sentiment_Momentum'].fillna(0)

    # 2. Target: 5-Day Future Return
    LOOK_AHEAD = 5
    df['Future_Close'] = df['Close'].shift(-LOOK_AHEAD)
    df['Future_Return'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    # 3. Create 3 Classes (0=Sell, 1=Hold, 2=Buy)
    # Threshold: 1.5% move needed to justify a trade
    THRESHOLD = 0.015 
    
    conditions = [
        (df['Future_Return'] < -THRESHOLD),  # Sell
        (df['Future_Return'] > THRESHOLD)    # Buy
    ]
    choices = [0, 2] # 0 for Sell, 2 for Buy
    
    # Default is 1 (Hold/Neutral)
    df['Target_Class'] = np.select(conditions, choices, default=1)
    
    # ... (Keep your other technical indicators like RSI, Volatility) ...
    
    # Drop NaNs created by the 5-day shift
    df_clean = df.dropna()
    
    return df_clean


def create_sequences(features, targets, seq_len):
    X = []
    y = []
    
    # Stop one step earlier because we need the NEXT value for the target
    for i in range(len(features) - seq_len):
        # Input: Sequence of length seq_len (e.g., Day 0 to Day 29)
        X.append(features[i:i+seq_len])
        
        # Target: The value at index i + seq_len (e.g., Day 30)
        # This represents the outcome happening AFTER the sequence ends
        y.append(targets[i+seq_len]) 
        
    return np.array(X), np.array(y)
def plot_training_loss(train_losses, val_losses):
    """
    Plots training and validation loss.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True)
    return plt

def load_scaler(scaler_type='feature'):
    """
    Loads a saved scaler.
    """
    path = f'{SCALER_PATH}{scaler_type}_scaler.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"Warning: Scaler not found at {path}")
        return None
