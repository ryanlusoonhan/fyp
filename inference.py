import json
import os

import joblib
import pandas as pd
import torch

from src.config import *
from src.model import LSTMModel


def predict_next_direction(model, recent_data, device):
    """Return (predicted_class, prob_up) for the next period."""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    pred_class = int(probs.argmax())
    prob_up = float(probs[1])
    return pred_class, prob_up


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    config_path = f"{MODEL_PATH}model_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config.json not found. Run train.py first.")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    feature_cols = config_data["feature_cols"]
    input_dim = config_data["input_dim"]
    seq_len = int(config_data.get("seq_len", SEQ_LEN))
    num_classes = int(config_data.get("num_classes", 2))

    if config_data.get("task") not in {None, "directional_classification"}:
        raise ValueError(
            "model_config.json is not for directional classification. "
            "Run train.py to regenerate classification artifacts."
        )

    scaler_path = f"{SCALER_PATH}feature_scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("feature_scaler.pkl not found. Run train.py first.")
    feature_scaler = joblib.load(scaler_path)

    data_path = f"{PROCESSED_DATA_PATH}training_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df_raw = pd.read_csv(data_path)
    if "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    missing_cols = [c for c in feature_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            "Feature columns in model config are missing from training_data.csv: "
            f"{missing_cols}. Run train.py to regenerate consistent artifacts."
        )

    df_recent = df_raw.tail(seq_len).copy()
    if len(df_recent) < seq_len:
        raise ValueError(f"Not enough data. Need {seq_len} rows, got {len(df_recent)}")

    last_actual_close = float(df_recent["Close"].iloc[-1]) if "Close" in df_recent.columns else None
    scaled_recent = feature_scaler.transform(df_recent[feature_cols].values)

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        output_dim=num_classes,
    ).to(DEVICE)

    model_path = f"{MODEL_PATH}best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("best_model.pth not found. Run train.py first.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError as exc:
        raise RuntimeError(
            "Model checkpoint shape does not match classification architecture. "
            "Run train.py to regenerate best_model.pth."
        ) from exc

    pred_class, prob_up = predict_next_direction(model, scaled_recent, DEVICE)
    direction = "UP" if pred_class == 1 else "DOWN_OR_FLAT"

    print("-" * 36)
    print("DIRECTIONAL PREDICTION REPORT")
    print("-" * 36)
    if last_actual_close is not None:
        print(f"Last Close:         ${last_actual_close:.2f}")
    print(f"Predicted Class:    {pred_class} ({direction})")
    print(f"Probability(UP):    {prob_up:.2%}")
    print(f"Probability(DOWN):  {1.0 - prob_up:.2%}")
    print("-" * 36)
