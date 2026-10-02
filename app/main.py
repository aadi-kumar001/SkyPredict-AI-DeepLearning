from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.weather import geocode_city, get_forecast
from app.ml.predict import predict_next_24h, model_available

app = FastAPI(
    title="AI Weather Intelligence API",
    version="1.0.0",
    description="Live weather + PyTorch LSTM forecasting service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_available": model_available(),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/weather")
def weather(city: str = Query(..., min_length=2)):
    try:
        location = geocode_city(city)
        data = get_forecast(
            location["latitude"],
            location["longitude"],
            location["timezone"],
        )

        ml_prediction = predict_next_24h(data["hourly"])

        hourly = data["hourly"]
        response = {
            "location": location,
            "current": data.get("current", {}),
            "current_units": data.get("current_units", {}),
            "hourly": {
                "time": hourly["time"][:48],
                "temperature_2m": hourly["temperature_2m"][:48],
                "relative_humidity_2m": hourly["relative_humidity_2m"][:48],
                "precipitation": hourly["precipitation"][:48],
                "wind_speed_10m": hourly["wind_speed_10m"][:48],
            },
            "ml_forecast": {
                "horizon_hours": 24,
                "temperature_c": ml_prediction,
            },
            "model": "PyTorch LSTM" if ml_prediction is not None else "Not trained yet",
            "source": "Open-Meteo",
        }
        return response

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather service error: {e}")
