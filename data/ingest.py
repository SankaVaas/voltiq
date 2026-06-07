"""
data/ingest.py — fetch electricity load and weather data from public APIs.

Data sources:
  - ENTSO-E Transparency Platform (EU grid load, generation mix)
  - Open-Meteo (weather: temperature, solar irradiance, wind speed)

All data is written as Parquet to data/raw/ for downstream processing.
CPU-friendly: no GPU required.
"""

from __future__ import annotations

from datetime import UTC, datetime

import openmeteo_requests
import pandas as pd
import requests_cache
from entsoe import EntsoePandasClient
from retry_requests import retry

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

RAW_DIR = settings.data_raw_dir
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── ENTSO-E ──────────────────────────────────────────────────────────────────

COUNTRY_CODES = {
    "DE": "10Y1001A1001A83F",
    "FR": "10YFR-RTE------C",
    "ES": "10YES-REE------0",
    "NL": "10YNL----------L",
    "PL": "10YPL-AREA-----S",
}


def fetch_entso_load(
    country: str = "DE",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch actual total load from ENTSO-E for a given country and time range.

    Returns a DataFrame with columns: [timestamp, country, load_mw]
    """
    if not settings.entso_e_api_key:
        logger.warning("ENTSO-E API key not set — returning synthetic load data")
        return _synthetic_load(country, start, end)

    client = EntsoePandasClient(api_key=settings.entso_e_api_key)

    start = start or datetime(2023, 1, 1, tzinfo=UTC)
    end = end or datetime(2024, 1, 1, tzinfo=UTC)

    country_code = COUNTRY_CODES.get(country, country)
    logger.info("Fetching ENTSO-E load", country=country, start=str(start), end=str(end))

    series = client.query_load(country_code, start=pd.Timestamp(start), end=pd.Timestamp(end))

    df = series.reset_index()
    df.columns = ["timestamp", "load_mw"]
    df["country"] = country
    df["load_mw"] = df["load_mw"].astype(float)

    out_path = RAW_DIR / f"entso_load_{country}_{start.date()}_{end.date()}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved ENTSO-E load data", path=str(out_path), rows=len(df))
    return df


def _synthetic_load(
    country: str,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    """Generate realistic synthetic load data when API key is unavailable."""
    import numpy as np

    start = start or datetime(2023, 1, 1)
    end = end or datetime(2024, 1, 1)
    rng = pd.date_range(start=start, end=end, freq="h")

    np.random.seed(42)
    base = 45_000
    hours = pd.Series(rng.hour)
    day_pattern = 8_000 * np.sin((hours - 6) * np.pi / 12).clip(0)
    week_pattern = np.where(rng.weekday >= 5, -5_000, 0)
    noise = np.random.normal(0, 1_500, len(rng))
    seasonal = 10_000 * np.cos((rng.dayofyear / 365) * 2 * np.pi)

    load = base + day_pattern.values + week_pattern + noise + seasonal

    df = pd.DataFrame({"timestamp": rng, "load_mw": load, "country": country})
    logger.info("Generated synthetic load data", rows=len(df), country=country)
    return df


# ── Open-Meteo ───────────────────────────────────────────────────────────────

CITY_COORDS = {
    "DE": (52.52, 13.41),
    "FR": (48.86, 2.35),
    "ES": (40.42, -3.70),
    "NL": (52.37, 4.90),
    "PL": (52.23, 21.01),
}


def fetch_weather(
    country: str = "DE",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo (free, no key required).

    Returns DataFrame with columns:
      [timestamp, temperature_2m, wind_speed_10m, shortwave_radiation, country]
    """
    cache_session = requests_cache.CachedSession(".weather_cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    start = start or datetime(2023, 1, 1)
    end = end or datetime(2024, 1, 1)
    lat, lon = CITY_COORDS.get(country, (52.52, 13.41))

    logger.info("Fetching Open-Meteo weather", country=country, lat=lat, lon=lon)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        "timezone": "UTC",
    }

    responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    response = responses[0]
    hourly = response.Hourly()

    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
        "shortwave_radiation": hourly.Variables(2).ValuesAsNumpy(),
        "country": country,
    })

    out_path = RAW_DIR / f"weather_{country}_{start.date()}_{end.date()}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved weather data", path=str(out_path), rows=len(df))
    return df


# ── Combined dataset ──────────────────────────────────────────────────────────

def build_feature_dataset(
    country: str = "DE",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Merge load + weather into a single feature DataFrame ready for model training.

    Adds temporal features: hour_of_day, day_of_week, month, is_weekend.
    Saves to data/processed/.
    """
    load_df = fetch_entso_load(country=country, start=start, end=end)
    weather_df = fetch_weather(country=country, start=start, end=end)

    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"]).dt.tz_localize(None)
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"]).dt.tz_localize(None)

    df = pd.merge(load_df, weather_df.drop(columns=["country"]), on="timestamp", how="inner")

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)

    out_dir = settings.data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"features_{country}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Built feature dataset", path=str(out_path), shape=df.shape)
    return df