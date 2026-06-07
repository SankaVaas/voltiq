"""api/routers/forecast.py"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from fastapi import APIRouter

from api.schemas import ForecastRequest, ForecastResponse

router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest) -> ForecastResponse:
    horizon = request.horizon_hours
    now = datetime.utcnow()
    timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(horizon)]
    base = 47_000
    hours = np.arange(horizon)
    p50 = (
        base
        + 6000 * np.sin((hours % 24 - 6) * np.pi / 12).clip(0)
        + np.random.normal(0, 800, horizon)
    )
    p10 = (p50 - 3000).tolist()
    p90 = (p50 + 3000).tolist()
    return ForecastResponse(
        country=request.country,
        horizon_hours=horizon,
        timestamps=timestamps,
        p10=p10,
        p50=p50.tolist(),
        p90=p90,
        peak_forecast_mw=max(p90),
        model_version="statistical_baseline_v0",
    )
