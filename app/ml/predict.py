import json
from pathlib import Path

import numpy as np
import torch

from app.ml.model import WeatherLSTM

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models" / "weather_lstm.pt"
SCALER_PATH = BASE / "models" / "scaler.json"


def model_available():
    return MODEL_PATH.exists() and SCALER_PATH.exists()


def predict_next_24h(hourly: dict):
    if not model_available():
        return None

    features = ["temperature_2m", "relative_humidity_2m", "pressure_msl",
                "wind_speed_10m", "precipitation"]

    values = np.column_stack([hourly[f] for f in features]).astype(np.float32)

    # Need the latest 168 valid hourly rows.
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 168:
        return None

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = WeatherLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        output_size=ckpt["horizon"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = json.loads(SCALER_PATH.read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)

    x = (values[-168:] - mean) / scale
    x = torch.tensor(x[None, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(x).numpy()[0]

    # Temperature is feature 0, so inverse transform only that feature.
    pred = pred_scaled * scale[0] + mean[0]
    return pred.tolist()
