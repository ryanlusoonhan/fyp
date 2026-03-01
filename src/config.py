import torch
import os


def _device_log(message: str):
    quiet = os.getenv("NELL_QUIET_DEVICE", "").lower() in {"1", "true", "yes"}
    if not quiet:
        print(message)

# -------------------
# DEVICE CONFIGURATION
# -------------------
# Check for CUDA (Windows) or MPS (Mac)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    _device_log(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    _device_log("Using Apple Silicon MPS")
else:
    DEVICE = torch.device("cpu")
    _device_log("Using CPU")

# -------------------
# DATA PARAMETERS
# -------------------
SEQ_LEN = 30         # Lookback window
TRAIN_SPLIT = 0.8    # 80% training data
VAL_SPLIT = 0.1      # 10% validation data

# -------------------
# MODEL PARAMETERS
# -------------------
BATCH_SIZE = 128      # Larger batch size = faster training on GPU
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_DIM = 16
NUM_LAYERS = 1
DROPOUT = 0.2
NHEAD = 8            # Attention heads (Transformer only)

# -------------------
# SENTIMENT PARAMETERS
# -------------------
FINBERT_MODEL = "ProsusAI/finbert"
SENTIMENT_WINDOW = 7

# -------------------
# TECHNICAL INDICATORS
# -------------------
RSI_PERIOD = 14
MA_PERIODS = [5, 10, 20, 50]

# -------------------
# PATHS
# -------------------
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "models/"
SCALER_PATH = "data/processed/scalers/"

# -------------------
# OPENBB SETTINGS
# -------------------
OPENBB_PROVIDER = "yfinance"
OPENBB_BATCH_START_DATE = "2015-01-01"

OPENBB_INDEX_SYMBOLS = ["^HSI", "^VIX", "^TNX", "^GSPC"]
OPENBB_EQUITY_SYMBOLS = [
    "0005.HK",
    "0700.HK",
    "0939.HK",
    "0941.HK",
    "1299.HK",
    "1398.HK",
    "1810.HK",
    "2318.HK",
    "3690.HK",
    "9988.HK",
]
OPENBB_CURRENCY_SYMBOLS = ["USDHKD", "USDCNY"]

OPENBB_RAW_DATA_PATH = "data/raw/openbb/"
OPENBB_INDEX_HISTORY_FILE = f"{OPENBB_RAW_DATA_PATH}index_history.csv"
OPENBB_EQUITY_HISTORY_FILE = f"{OPENBB_RAW_DATA_PATH}equity_history.csv"
OPENBB_CURRENCY_HISTORY_FILE = f"{OPENBB_RAW_DATA_PATH}currency_history.csv"
OPENBB_NEWS_FILE = f"{OPENBB_RAW_DATA_PATH}news_recent.csv"
OPENBB_MANIFEST_FILE = f"{OPENBB_RAW_DATA_PATH}manifest.json"
OPENBB_TRAINING_FILE = f"{PROCESSED_DATA_PATH}training_data_openbb.csv"
OPENBB_REFRESH_STATUS_FILE = f"{MODEL_PATH}openbb_refresh_status.json"
SIGNAL_HISTORY_FILE = f"{MODEL_PATH}signal_history_weekly.csv"
