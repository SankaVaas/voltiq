"""api/routers/ingest.py"""
from datetime import datetime, timedelta

from fastapi import APIRouter

from api.schemas import IngestRequest, IngestResponse
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingest(request: IngestRequest) -> IngestResponse:
    from data.ingest import build_feature_dataset

    try:
        end = datetime.utcnow()
        start = end - timedelta(days=request.days_back)
        df = build_feature_dataset(country=request.country, start=start, end=end)
        return IngestResponse(
            status="ok",
            country=request.country,
            rows_ingested=len(df),
            latest_timestamp=str(df["timestamp"].max()),
        )
    except Exception as e:
        logger.error("Ingest failed", error=str(e))
        return IngestResponse(
            status="error",
            country=request.country,
            rows_ingested=0,
            error=str(e),
        )