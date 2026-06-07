"""api/routers/anomalies.py"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Query

from anomaly.detector import AnomalyDetector
from api.schemas import AnomalyResponse

router = APIRouter()


@router.get("/anomalies", response_model=AnomalyResponse)
async def get_anomalies(country: str = Query(default="DE")) -> AnomalyResponse:
    rng = np.random.default_rng(0)
    series = rng.normal(50_000, 2_000, 168)
    series[72:76] *= 1.4

    detector = AnomalyDetector(threshold_percentile=95.0)
    detector.train(series, epochs=5)
    result = detector.detect(series)

    severity = "high" if result["anomaly_rate"] > 0.05 else "low"
    return AnomalyResponse(
        country=country,
        anomaly_count=len(result["anomaly_indices"]),
        anomaly_rate=result["anomaly_rate"],
        severity=severity,
        threshold=result["threshold"],
        anomaly_indices=result["anomaly_indices"][:20],
    )