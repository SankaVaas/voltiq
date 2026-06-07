"""mlflow_tracking/tracker.py — MLflow helpers for experiment tracking."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mlflow
import mlflow.pytorch
import torch.nn as nn

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


@contextmanager
def forecast_run(run_name: str, tags: dict[str, str] | None = None) -> Generator:
    mlflow.set_experiment(settings.mlflow_experiment_forecast)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        logger.info("MLflow forecast run started", run_id=run.info.run_id)
        yield run
        logger.info("MLflow forecast run ended", run_id=run.info.run_id)


@contextmanager
def anomaly_run(run_name: str, tags: dict[str, str] | None = None) -> Generator:
    mlflow.set_experiment(settings.mlflow_experiment_anomaly)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        logger.info("MLflow anomaly run started", run_id=run.info.run_id)
        yield run
        logger.info("MLflow anomaly run ended", run_id=run.info.run_id)


def log_forecast_metrics(
    mae: float,
    rmse: float,
    mape: float,
    pinball_loss_p10: float,
    pinball_loss_p90: float,
    step: int | None = None,
) -> None:
    mlflow.log_metrics({
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pinball_p10": pinball_loss_p10,
        "pinball_p90": pinball_loss_p90,
    }, step=step)


def log_tft_params(config_dict: dict[str, Any]) -> None:
    mlflow.log_params(config_dict)


def register_forecast_model(model: nn.Module, model_name: str = "voltiq_tft") -> str:
    model_uri = mlflow.pytorch.log_model(model, artifact_path="model").model_uri
    registered = mlflow.register_model(model_uri, model_name)
    logger.info("Model registered", name=model_name, version=registered.version)
    return registered.version


def get_production_model(model_name: str = "voltiq_tft") -> nn.Module | None:
    try:
        model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
        logger.info("Loaded production model", name=model_name)
        return model  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load production model", name=model_name, error=str(e))
        return None


def compare_runs(experiment_name: str, metric: str = "mae", n: int = 5) -> list[dict[str, Any]]:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=n,
    )
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            metric: r.data.metrics.get(metric),
        }
        for r in runs
    ]