"""api/schemas.py — Pydantic v2 request/response schemas for all endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=2000, description="Natural language operator query"
    )
    country: str = Field(default="DE", description="ISO country code (DE, FR, ES, NL, PL)")
    include_forecast: bool = Field(default=True)
    include_anomaly: bool = Field(default=True)

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        allowed = {"DE", "FR", "ES", "NL", "PL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"country must be one of {allowed}")
        return upper


class QueryResponse(BaseModel):
    response: str
    query_type: Literal["forecast", "anomaly", "qa", "general"]
    country: str
    forecast: dict[str, Any] | None = None
    anomalies: dict[str, Any] | None = None
    processing_time_ms: float


class ForecastRequest(BaseModel):
    country: str = Field(default="DE")
    horizon_hours: int = Field(default=48, ge=1, le=168)


class ForecastResponse(BaseModel):
    country: str
    horizon_hours: int
    timestamps: list[str]
    p10: list[float]
    p50: list[float]
    p90: list[float]
    peak_forecast_mw: float
    model_version: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class AnomalyResponse(BaseModel):
    country: str
    anomaly_count: int
    anomaly_rate: float
    severity: Literal["low", "medium", "high"]
    threshold: float
    anomaly_indices: list[int]
    scanned_at: datetime = Field(default_factory=datetime.utcnow)


class IngestRequest(BaseModel):
    country: str = Field(default="DE")
    days_back: int = Field(default=7, ge=1, le=365)


class IngestResponse(BaseModel):
    status: Literal["ok", "error"]
    country: str
    rows_ingested: int
    latest_timestamp: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    environment: str
    services: dict[str, str]