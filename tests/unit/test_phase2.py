"""tests/unit/test_phase2.py — Phase 2: training, metrics, CLI unit tests."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── GridSequenceDataset ───────────────────────────────────────────────────────

class TestGridSequenceDataset:
    def _make_df(self, n: int = 400) -> object:
        import pandas as pd
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "timestamp":          pd.date_range("2023-01-01", periods=n, freq="h"),
            "load_mw":            rng.normal(50_000, 2_000, n),
            "temperature_2m":     rng.normal(12, 5, n),
            "wind_speed_10m":     np.abs(rng.normal(5, 2, n)),
            "shortwave_radiation": np.abs(rng.normal(100, 80, n)),
            "hour_of_day":        pd.date_range("2023-01-01", periods=n, freq="h").hour,
            "day_of_week":        pd.date_range("2023-01-01", periods=n, freq="h").dayofweek,
            "month":              pd.date_range("2023-01-01", periods=n, freq="h").month,
            "country_id":         [0] * n,
            "is_weekend":         [0] * n,
        })

    def test_length(self) -> None:
        from forecasting.training.train import GridSequenceDataset
        df = self._make_df(300)
        ds = GridSequenceDataset(df, encoder_length=48, decoder_length=12)
        assert len(ds) == 300 - 48 - 12 + 1

    def test_item_shapes(self) -> None:
        from forecasting.training.train import CATEGORICAL_COLS, NUMERIC_COLS, GridSequenceDataset
        df = self._make_df(300)
        ds = GridSequenceDataset(df, encoder_length=48, decoder_length=12)
        item = ds[0]
        assert item["enc_numeric"].shape    == (48, len(NUMERIC_COLS))
        assert item["enc_categorical"].shape == (48, len(CATEGORICAL_COLS))
        assert item["dec_numeric"].shape    == (12, len(NUMERIC_COLS))
        assert item["dec_categorical"].shape == (12, len(CATEGORICAL_COLS))
        assert item["target"].shape         == (12,)

    def test_normalisation(self) -> None:
        from forecasting.training.train import GridSequenceDataset
        df = self._make_df(300)
        ds = GridSequenceDataset(df, encoder_length=48, decoder_length=12)
        # Numeric data should be normalised (mean ~0, std ~1)
        assert ds.numeric.mean().abs() < 1.0
        assert ds.numeric.std().item() < 3.0

    def test_too_short_raises(self) -> None:
        from forecasting.training.train import GridSequenceDataset
        df = self._make_df(10)
        with pytest.raises(AssertionError):
            GridSequenceDataset(df, encoder_length=48, decoder_length=12)


# ── Quantile loss ─────────────────────────────────────────────────────────────

class TestQuantileLoss:
    def test_loss_positive(self) -> None:
        from forecasting.training.train import quantile_loss
        preds = torch.randn(8, 24, 3)
        tgt   = torch.randn(8, 24)
        loss  = quantile_loss(preds, tgt, [0.1, 0.5, 0.9])
        assert loss.item() > 0

    def test_perfect_median_low_loss(self) -> None:
        from forecasting.training.train import quantile_loss
        tgt   = torch.ones(4, 12) * 50_000
        preds = torch.zeros(4, 12, 3)
        preds[:, :, 0] = 47_000   # p10
        preds[:, :, 1] = 50_000   # p50 = exact
        preds[:, :, 2] = 53_000   # p90
        loss = quantile_loss(preds, tgt, [0.1, 0.5, 0.9])
        assert loss.item() < 1_000  # very low loss when p50 is exact

    def test_returns_scalar(self) -> None:
        from forecasting.training.train import quantile_loss
        preds = torch.randn(2, 6, 3)
        tgt   = torch.randn(2, 6)
        loss  = quantile_loss(preds, tgt, [0.1, 0.5, 0.9])
        assert loss.ndim == 0


# ── Compute metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_zero_error(self) -> None:
        from forecasting.training.train import compute_metrics
        tgt = torch.ones(10, 24) * 50_000
        metrics = compute_metrics(tgt, tgt)
        assert metrics["mae"]  < 1.0
        assert metrics["rmse"] < 1.0
        assert metrics["mape"] < 0.01

    def test_keys_present(self) -> None:
        from forecasting.training.train import compute_metrics
        tgt  = torch.randn(4, 12) * 1_000 + 50_000
        pred = tgt + torch.randn(4, 12) * 500
        m = compute_metrics(pred, tgt)
        assert set(m.keys()) == {"mae", "rmse", "mape"}

    def test_mae_positive(self) -> None:
        from forecasting.training.train import compute_metrics
        tgt  = torch.ones(4, 12) * 50_000
        pred = tgt + 1_000
        m = compute_metrics(pred, tgt)
        assert m["mae"] > 0


# ── ForecastMetrics ───────────────────────────────────────────────────────────

class TestForecastMetrics:
    def _arrays(self, n: int = 168) -> tuple:
        rng = np.random.default_rng(1)
        actual = rng.normal(50_000, 3_000, n)
        p50    = actual + rng.normal(0, 1_500, n)
        p10    = p50 - 3_000
        p90    = p50 + 3_000
        return actual, p10, p50, p90

    def test_evaluate_returns_metrics(self) -> None:
        from forecasting.evaluation.metrics import evaluate
        actual, p10, p50, p90 = self._arrays()
        m = evaluate(actual, p10, p50, p90)
        assert m.mae > 0
        assert m.rmse >= m.mae
        assert 0.0 <= m.coverage_80 <= 1.0
        assert m.winkler_80 > 0

    def test_to_dict_has_all_keys(self) -> None:
        from forecasting.evaluation.metrics import evaluate
        actual, p10, p50, p90 = self._arrays()
        d = evaluate(actual, p10, p50, p90).to_dict()
        expected = {"mae", "rmse", "mape", "smape", "pinball_p10",
                    "pinball_p50", "pinball_p90", "coverage_80", "winkler_80"}
        assert set(d.keys()) == expected

    def test_perfect_forecast(self) -> None:
        from forecasting.evaluation.metrics import evaluate
        actual = np.ones(100) * 50_000
        m = evaluate(actual, actual - 1, actual, actual + 1)
        assert m.mae < 1.0
        assert m.coverage_80 == 1.0

    def test_baseline_worse_than_tight_forecast(self) -> None:
        from forecasting.evaluation.metrics import evaluate, evaluate_baseline
        rng = np.random.default_rng(2)
        actual = rng.normal(50_000, 2_000, 200)
        p50    = actual + rng.normal(0, 500, 200)
        p10, p90 = p50 - 1_000, p50 + 1_000
        model_m    = evaluate(actual, p10, p50, p90)
        baseline_m = evaluate_baseline(actual)
        assert model_m.mae < baseline_m.mae

    def test_pinball_loss_direct(self) -> None:
        from forecasting.evaluation.metrics import pinball_loss
        actual = np.array([100.0, 200.0, 300.0])
        pred   = np.array([100.0, 200.0, 300.0])
        assert pinball_loss(actual, pred, 0.5) < 1e-6  # perfect median

    def test_winkler_wider_interval_lower_score(self) -> None:
        from forecasting.evaluation.metrics import winkler_score
        actual = np.ones(50) * 50_000
        # Perfect interval (no misses, width=0 at exact values)
        tight_score  = winkler_score(actual, actual, actual, alpha=0.2)
        # Wide interval (no misses, but wide)
        wide_score   = winkler_score(actual, actual - 10_000, actual + 10_000, alpha=0.2)
        assert wide_score > tight_score


# ── Train smoke test ──────────────────────────────────────────────────────────

class TestTrainSmoke:
    def test_fast_dev_run(self) -> None:
        """Verify the full train() function runs without error in fast-dev mode."""
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_artifact"),
        ):
            mock_run.return_value.__enter__ = MagicMock(
                return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
            )
            mock_run.return_value.__exit__ = MagicMock(return_value=False)

            from forecasting.training.train import train
            result = train(country="DE", fast_dev=True)

        assert "best_val_mae" in result
        assert "best_epoch" in result
        assert result["best_val_mae"] >= 0


# ── Airflow task callables ────────────────────────────────────────────────────

class TestAirflowTasks:
    def _mock_context(self) -> dict:
        ti = MagicMock()
        ti.xcom_push = MagicMock()
        ti.xcom_pull = MagicMock(return_value=None)
        return {
            "task_instance": ti,
            "execution_date": datetime(2024, 1, 15, 2, 0),
        }

    def test_ingest_task_runs(self) -> None:
        from pipelines.dags.voltiq_pipeline import task_ingest_data
        ctx = self._mock_context()
        task_ingest_data(**ctx)
        ctx["task_instance"].xcom_push.assert_called_once()

    def test_preprocess_task_runs(self) -> None:
        from pipelines.dags.voltiq_pipeline import task_preprocess_data
        ctx = self._mock_context()
        task_preprocess_data(**ctx)  # should not raise even with no files

    def test_notify_task_runs(self) -> None:
        from pipelines.dags.voltiq_pipeline import task_notify
        ctx = self._mock_context()
        task_notify(**ctx)  # should not raise