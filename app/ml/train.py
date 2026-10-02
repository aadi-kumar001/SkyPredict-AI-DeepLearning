import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import WeatherLSTM
from app.weather import FEATURES, get_historical

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

LAT = float(os.getenv("TRAIN_LAT", "28.6139"))
LON = float(os.getenv("TRAIN_LON", "77.2090"))

HISTORY_HOURS = 168
HORIZON = 24
EPOCHS = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE = 64


def main():
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=730)

    print(f"Downloading historical data: {start} -> {end}")
    data = get_historical(LAT, LON, start.isoformat(), end.isoformat(), "auto")

    hourly = data["hourly"]
    columns = [hourly[k] for k in FEATURES]
    arr = np.column_stack(columns).astype(np.float32)

    # Remove rows with missing values.
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]

    if len(arr) < HISTORY_HOURS + HORIZON + 100:
        raise RuntimeError("Not enough historical observations for training.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)

    X, y = [], []
    for i in range(HISTORY_HOURS, len(scaled) - HORIZON):
        X.append(scaled[i - HISTORY_HOURS:i])
        # Temperature is feature 0.
        y.append(scaled[i:i + HORIZON, 0])

    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.float32)

    # Chronological split: do not shuffle across train/test.
    split = int(len(X) * 0.85)
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = WeatherLSTM(input_size=len(FEATURES), output_size=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)

        train_loss = total / len(train_ds)

        model.eval()
        with torch.no_grad():
            test_pred = model(X[split:])
            test_loss = loss_fn(test_pred, y[split:]).item()

        print(f"Epoch {epoch:02d}/{EPOCHS} | train={train_loss:.5f} | test={test_loss:.5f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": len(FEATURES),
            "hidden_size": 64,
            "num_layers": 2,
            "horizon": HORIZON,
            "features": FEATURES,
        },
        MODEL_DIR / "weather_lstm.pt",
    )

    scaler_payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": FEATURES,
    }
    (MODEL_DIR / "scaler.json").write_text(json.dumps(scaler_payload, indent=2))

    print(f"Saved model: {MODEL_DIR / 'weather_lstm.pt'}")
    print(f"Saved scaler: {MODEL_DIR / 'scaler.json'}")


if __name__ == "__main__":
    main()
