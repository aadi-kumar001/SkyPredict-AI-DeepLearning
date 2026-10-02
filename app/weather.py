import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
]


def geocode_city(city: str):
    r = requests.get(
        GEOCODE_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=20,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}")
    x = results[0]
    return {
        "name": x["name"],
        "country": x.get("country", ""),
        "latitude": x["latitude"],
        "longitude": x["longitude"],
        "timezone": x.get("timezone", "auto"),
    }


def get_forecast(lat: float, lon: float, timezone="auto"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(FEATURES),
        "current": ",".join(FEATURES[:4]),
        "forecast_days": 7,
        "timezone": timezone,
        "wind_speed_unit": "kmh",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def get_historical(lat: float, lon: float, start_date: str, end_date: str, timezone="auto"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(FEATURES),
        "timezone": timezone,
        "wind_speed_unit": "kmh",
    }
    r = requests.get(HISTORICAL_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()
