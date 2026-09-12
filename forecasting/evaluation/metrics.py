"""
forecasting/evaluation/metrics.py — Evaluation metrics for grid load forecasting.

Metrics:
  - MAE   : Mean Absolute Error (MW) — intuitive error in megawatts
  - RMSE  : Root Mean Squared Error — penalises large errors more
  - MAPE  : Mean Absolute Percentage Error — scale-independent
  - sMAPE : Symmetric MAPE — avoids asymmetry near zero
  - Pinball loss per quantile — proper scoring rule for probabilistic forecasts
  - Coverage: what fraction of actuals fall within the p10–p90 interval
  - Winkler score: rewards sharp, calibrated intervals
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    coverage_80: float    # fraction of actuals inside p10–p90
    winkler_80: float     # Winkler interval score (lower = better)

    def to_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "smape": self.smape,
            "pinball_p10": self.pinball_p10,
            "pinball_p50": self.pinball_p50,
            "pinball_p90": self.pinball_p90,
            "coverage_80": self.coverage_80,
            "winkler_80": self.winkler_80,
        }

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:,.1f} MW  RMSE={self.rmse:,.1f} MW  "
            f"MAPE={self.mape:.2f}%  Coverage={self.coverage_80:.1%}  "
            f"Winkler={self.winkler_80:,.1f}"
        )


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    """Pinball / quantile loss. Lower is better."""
    errors = actual - predicted
    return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))


def winkler_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """
    Winkler interval score for (1-alpha) prediction interval.
    Penalises wide intervals and misses. Lower is better.
    """
    width = upper - lower
    miss_low  = 2 / alpha * np.maximum(lower - actual, 0)
    miss_high = 2 / alpha * np.maximum(actual - upper, 0)
    return float(np.mean(width + miss_low + miss_high))


def evaluate(
    actual: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> ForecastMetrics:
    """
    Compute all forecasting metrics.

    Args:
        actual: observed load values (MW)
        p10:    10th percentile forecast
        p50:    median forecast
        p90:    90th percentile forecast

    Returns:
        ForecastMetrics dataclass
    """
    actual = np.asarray(actual, dtype=float)
    p10    = np.asarray(p10,    dtype=float)
    p50    = np.asarray(p50,    dtype=float)
    p90    = np.asarray(p90,    dtype=float)

    errors = actual - p50
    abs_errors = np.abs(errors)

    mae  = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(abs_errors / (np.abs(actual) + 1e-8)) * 100)
    smape = float(np.mean(
        2 * abs_errors / (np.abs(actual) + np.abs(p50) + 1e-8)
    ) * 100)

    pb10 = pinball_loss(actual, p10, 0.1)
    pb50 = pinball_loss(actual, p50, 0.5)
    pb90 = pinball_loss(actual, p90, 0.9)

    coverage = float(np.mean((actual >= p10) & (actual <= p90)))
    winkler  = winkler_score(actual, p10, p90, alpha=0.2)

    return ForecastMetrics(
        mae=mae, rmse=rmse, mape=mape, smape=smape,
        pinball_p10=pb10, pinball_p50=pb50, pinball_p90=pb90,
        coverage_80=coverage, winkler_80=winkler,
    )


def evaluate_baseline(
    actual: np.ndarray,
    seasonal_period: int = 168,
) -> ForecastMetrics:
    """
    Evaluate a naive seasonal baseline (repeat last week's values).
    Useful for benchmarking TFT improvement over a simple baseline.
    """
    n = len(actual)
    p50 = np.array([actual[max(0, i - seasonal_period)] for i in range(n)])
    noise = np.std(actual) * 0.15
    p10 = p50 - 1.28 * noise
    p90 = p50 + 1.28 * noise
    return evaluate(actual, p10, p50, p90)


def log_metrics_to_mlflow(metrics: ForecastMetrics, step: int | None = None) -> None:
    """Log ForecastMetrics to the active MLflow run."""
    import mlflow
    mlflow.log_metrics(metrics.to_dict(), step=step)