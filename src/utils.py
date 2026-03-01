import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Feature engineering (PAST-ONLY) --------
def engineer_features_past_only(df: pd.DataFrame, sentiment_window: int = 7) -> pd.DataFrame:
    df = df.copy()
    
    # Sentiment smoothing (past-only rolling window)
    if "sentiment_score" in df.columns:
        df["Sentiment_MA7"] = df["sentiment_score"].rolling(window=sentiment_window, min_periods=1).mean()
        df["Sentiment_Momentum"] = df["sentiment_score"] - df["Sentiment_MA7"]
    else:
        df["Sentiment_MA7"] = 0.0
        df["Sentiment_Momentum"] = 0.0
        
    # Example: past-only returns/volatility (optional but safe)
    if "Close" in df.columns:
        df["Return_1D"] = df["Close"].pct_change().fillna(0.0)
        df["Volatility_10D"] = df["Return_1D"].rolling(10, min_periods=2).std().fillna(0.0)
        
    return df


def engineer_features_market_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = pd.to_numeric(df.get("Close", pd.Series(0.0, index=df.index)), errors="coerce").ffill().fillna(0.0)
    open_ = pd.to_numeric(df.get("Open", close), errors="coerce").fillna(close)
    high = pd.to_numeric(df.get("High", close), errors="coerce").fillna(close)
    low = pd.to_numeric(df.get("Low", close), errors="coerce").fillna(close)

    def safe_pct(series: pd.Series, periods: int = 1) -> pd.Series:
        return series.pct_change(periods).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["HSI_Return_1D"] = safe_pct(close, 1)
    df["HSI_Return_5D"] = safe_pct(close, 5)
    df["HSI_Return_20D"] = safe_pct(close, 20)
    df["HSI_Volatility_10D"] = df["HSI_Return_1D"].rolling(10, min_periods=2).std().fillna(0.0)
    df["HSI_Volatility_20D"] = df["HSI_Return_1D"].rolling(20, min_periods=5).std().fillna(0.0)
    df["HSI_Range_Pct"] = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["HSI_Gap_Pct"] = ((open_ - close.shift(1)) / close.shift(1).replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    vix = pd.to_numeric(df.get("VIX_Close", pd.Series(0.0, index=df.index)), errors="coerce").ffill().fillna(0.0)
    tnx = pd.to_numeric(df.get("TNX_Close", pd.Series(0.0, index=df.index)), errors="coerce").ffill().fillna(0.0)
    gspc = pd.to_numeric(df.get("GSPC_Close", pd.Series(0.0, index=df.index)), errors="coerce").ffill().fillna(0.0)
    usdhkd = pd.to_numeric(df.get("USDHKD_Close", pd.Series(0.0, index=df.index)), errors="coerce").ffill().fillna(0.0)
    usdcny = pd.to_numeric(df.get("USDCNY_Close", pd.Series(0.0, index=df.index)), errors="coerce").ffill().fillna(0.0)

    df["VIX_Change_1D"] = safe_pct(vix, 1)
    vix_mean = vix.rolling(20, min_periods=5).mean()
    vix_std = vix.rolling(20, min_periods=5).std().replace(0.0, np.nan)
    df["VIX_ZScore_20D"] = ((vix - vix_mean) / vix_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["TNX_Change_1D"] = safe_pct(tnx, 1)
    df["GSPC_Return_1D"] = safe_pct(gspc, 1)
    df["HSI_vs_GSPC_Return_Spread_1D"] = df["HSI_Return_1D"] - df["GSPC_Return_1D"]
    df["USDHKD_Change_1D"] = safe_pct(usdhkd, 1)
    df["USDCNY_Change_1D"] = safe_pct(usdcny, 1)

    for col in [
        "HK_Breadth_Positive_1D",
        "HK_Breadth_Above_MA20",
        "HK_Breadth_Dispersion_5D",
    ]:
        series = df[col] if col in df.columns else pd.Series(0.0, index=df.index)
        df[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)

    # Keep legacy aliases for compatibility with existing scripts/UI copy.
    df["Return_1D"] = df["HSI_Return_1D"]
    df["Volatility_10D"] = df["HSI_Volatility_10D"]

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

# -------- TRIPLE BARRIER LABELING (NEW) --------
def add_triple_barrier_labels(df, barrier_window=10, profit_take=0.03, stop_loss=0.015):
    """
    Creates Binary Target: 1 (Buy) vs 0 (Hold/Sell)
    
    Logic:
    - Label 1 (Buy): Price hits +3% (profit_take) BEFORE hitting -1.5% (stop_loss) or timeout.
    - Label 0 (Else): Price hits stop-loss OR doesn't move enough (Hold).
    """
    df = df.copy()
    close_prices = df["Close"].values
    labels = np.zeros(len(df)) # Default to 0
    
    # We need Future Return for Backtesting calculations later
    # (Close in 5 days - Today) / Today
    df["Future_Close"] = df["Close"].shift(-barrier_window)
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]

    for i in range(len(df) - barrier_window):
        current_price = close_prices[i]
        
        # Define Targets
        upper_barrier = current_price * (1 + profit_take)
        lower_barrier = current_price * (1 - stop_loss)
        
        # Get the window of future prices
        window = close_prices[i+1 : i+1+barrier_window]
        
        # Check when barriers are crossed
        hit_upper = np.where(window >= upper_barrier)[0]
        hit_lower = np.where(window <= lower_barrier)[0]
        
        first_upper = hit_upper[0] if len(hit_upper) > 0 else 999
        first_lower = hit_lower[0] if len(hit_lower) > 0 else 999
        
        # LOGIC: Did we hit Profit BEFORE Stop Loss and BEFORE Time Limit?
        if first_upper < first_lower and first_upper < barrier_window:
            labels[i] = 1 # BUY Signal
        else:
            labels[i] = 0 # NO BUY (Either Hold or Sell)
            
    df["Target_Class"] = labels
    
    # Drop the end rows where we can't calculate barriers
    df = df.dropna(subset=["Future_Close", "Future_Return"])
    
    return df

# -------- Sequence building --------
def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    X, y = [], []
    # Target at index i+seq_len corresponds to the day AFTER the input window ends
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)

# -------- Train/Val split that avoids overlap --------
def time_split_with_gap(df: pd.DataFrame, train_split: float, gap: int):
    """
    Splits by time index:
    train = [0 : train_end)
    val   = [train_end + gap : end)
    The gap prevents the last training window from sharing timesteps with the first validation window.
    """
    n = len(df)
    train_end = int(n * train_split)
    val_start = min(n, train_end + gap)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[val_start:].copy()
    
    return df_train, df_val

def plot_training_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Model Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    return plt
