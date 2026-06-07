"""api/routers/health.py — liveness and readiness probes."""

from fastapi import APIRouter
from qdrant_client import QdrantClient

from api.schemas import HealthResponse
from core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    services: dict[str, str] = {}

    # Qdrant
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=2)
        client.get_collections()
        services["qdrant"] = "ok"
    except Exception:
        services["qdrant"] = "unreachable"

    # MLflow (just check URL reachability)
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"{settings.mlflow_tracking_uri}/health")
            services["mlflow"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        services["mlflow"] = "unreachable"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.app_env,
        services=services,
    )
