"""api/routers/agents.py — agentic query endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from agents.graph import run_agent
from api.schemas import QueryRequest, QueryResponse
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def agent_query(request: QueryRequest) -> QueryResponse:
    t0 = time.monotonic()
    try:
        result = await run_agent(query=request.query, country=request.country)
    except Exception as e:
        logger.error("Agent query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    elapsed_ms = (time.monotonic() - t0) * 1000
    return QueryResponse(
        response=result["response"],
        query_type=result["query_type"],
        country=result["country"],
        forecast=result.get("forecast") if request.include_forecast else None,
        anomalies=result.get("anomalies") if request.include_anomaly else None,
        processing_time_ms=round(elapsed_ms, 2),
    )
