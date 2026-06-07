"""
api/main.py — FastAPI application entry point for Voltiq.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.middleware.logging import LoggingMiddleware
from api.middleware.metrics import MetricsMiddleware
from api.routers import agents, anomalies, forecast, health, ingest  # noqa: E501
from core.config import settings
from core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic."""
    setup_logging()
    logger.info("Voltiq API starting", version=settings.app_version, env=settings.app_env)
    yield
    logger.info("Voltiq API shutting down")


app = FastAPI(
    title="Voltiq API",
    description="Intelligent Renewable Energy Grid Analytics Platform",
    version=settings.app_version,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["https://voltiq.internal"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

app.include_router(health.router, tags=["Health"])
app.include_router(agents.router, prefix="/api/v1", tags=["Agent"])
app.include_router(forecast.router, prefix="/api/v1", tags=["Forecast"])
app.include_router(anomalies.router, prefix="/api/v1", tags=["Anomalies"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.app_env == "development",
    )