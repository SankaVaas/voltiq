"""tests/unit/test_core.py — unit tests for core modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from core.config import Settings

# ── Settings ──────────────────────────────────────────────────────────────────


class TestSettings:
    def test_default_values(self) -> None:
        s = Settings()
        assert s.app_name == "voltiq"
        assert s.forecast_horizon == 48
        assert s.forecast_lookback == 168

    def test_qdrant_url(self) -> None:
        s = Settings()
        assert s.qdrant_url == f"http://{s.qdrant_host}:{s.qdrant_port}"

    def test_log_level_uppercased(self) -> None:
        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_invalid_log_level(self) -> None:
        with pytest.raises(Exception):
            Settings(log_level="INVALID")

    def test_is_production_false(self) -> None:
        s = Settings(app_env="development")
        assert not s.is_production

    def test_is_production_true(self) -> None:
        s = Settings(app_env="production")
        assert s.is_production


# ── LSTM Autoencoder ──────────────────────────────────────────────────────────


class TestLSTMAutoencoder:
    def test_forward_shape(self) -> None:
        from anomaly.detector import LSTMAutoencoder

        model = LSTMAutoencoder(window_size=24, hidden_size=32, latent_dim=8, num_layers=1)
        x = torch.randn(4, 24, 1)
        recon = model(x)
        assert recon.shape == x.shape

    def test_reconstruction_error_shape(self) -> None:
        from anomaly.detector import LSTMAutoencoder

        model = LSTMAutoencoder(window_size=24, hidden_size=32, latent_dim=8, num_layers=1)
        x = torch.randn(8, 24, 1)
        err = model.reconstruction_error(x)
        assert err.shape == (8,)
        assert (err >= 0).all()

    def test_reconstruction_error_is_float_list(self) -> None:
        from anomaly.detector import LSTMAutoencoder

        model = LSTMAutoencoder(window_size=10, hidden_size=16, latent_dim=4, num_layers=1)
        x = torch.randn(3, 10, 1)
        err = model.reconstruction_error(x)
        as_list = err.tolist()
        assert isinstance(as_list, list)
        assert all(isinstance(v, float) for v in as_list)


# ── AnomalyDetector ───────────────────────────────────────────────────────────


class TestAnomalyDetector:
    def test_train_produces_losses(self) -> None:
        from anomaly.detector import AnomalyDetector

        rng = np.random.default_rng(0)
        series = rng.normal(50_000, 1_000, 200).astype(np.float32)
        det = AnomalyDetector(window_size=24, threshold_percentile=90.0)
        losses = det.train(series, epochs=3, batch_size=16)
        assert len(losses) == 3
        assert all(lo >= 0 for lo in losses)
        assert det.threshold is not None

    def test_detect_returns_expected_keys(self) -> None:
        from anomaly.detector import AnomalyDetector

        rng = np.random.default_rng(1)
        series = rng.normal(50_000, 1_000, 200).astype(np.float32)
        det = AnomalyDetector(window_size=24, threshold_percentile=90.0)
        det.train(series, epochs=2, batch_size=16)
        result = det.detect(series)
        assert "anomaly_indices" in result
        assert "anomaly_rate" in result
        assert "threshold" in result
        assert 0.0 <= result["anomaly_rate"] <= 1.0

    def test_detect_without_train_raises(self) -> None:
        from anomaly.detector import AnomalyDetector

        det = AnomalyDetector(window_size=10)
        series = np.random.rand(50).astype(np.float32)
        with pytest.raises(RuntimeError, match="Call train"):
            det.detect(series)

    def test_spike_flagged_as_anomaly(self) -> None:
        from anomaly.detector import AnomalyDetector

        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 300).astype(np.float32)
        det = AnomalyDetector(window_size=24, threshold_percentile=80.0)
        det.train(series, epochs=5, batch_size=32)
        test = rng.normal(0, 1, 100).astype(np.float32)
        test[50:54] *= 20.0  # large spike
        result = det.detect(test)
        assert len(result["anomaly_indices"]) > 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        from anomaly.detector import AnomalyDetector

        rng = np.random.default_rng(2)
        series = rng.normal(0, 1, 100).astype(np.float32)
        det = AnomalyDetector(window_size=10)
        det.train(series, epochs=2)
        path = tmp_path / "model.pt"
        det.save(path)
        det2 = AnomalyDetector(window_size=10, model_path=path)
        assert det2.threshold is not None
        assert abs(det2.threshold - det.threshold) < 1e-5  # type: ignore[arg-type]


# ── TFT model ─────────────────────────────────────────────────────────────────


class TestTFT:
    def test_forward_pass_shapes(self) -> None:
        from forecasting.models.tft import TemporalFusionTransformer, TFTConfig

        config = TFTConfig(
            num_numeric_features=4,
            num_categorical_features=2,
            categorical_vocab_sizes=[6, 2],
            hidden_size=32,
            lstm_layers=1,
            attention_heads=2,
            encoder_length=24,
            decoder_length=12,
            quantiles=[0.1, 0.5, 0.9],
        )
        model = TemporalFusionTransformer(config)
        batch = 2
        enc_num = torch.randn(batch, 24, 4)
        enc_cat = torch.randint(0, 2, (batch, 24, 2))
        dec_num = torch.randn(batch, 12, 4)
        dec_cat = torch.randint(0, 2, (batch, 12, 2))
        out = model(enc_num, enc_cat, dec_num, dec_cat)
        assert out["quantile_forecasts"].shape == (batch, 12, 3)
        assert out["attention_weights"].shape[0] == batch
        assert out["encoder_variable_weights"].shape == (batch, 24, 6)

    def test_output_keys_present(self) -> None:
        from forecasting.models.tft import TemporalFusionTransformer, TFTConfig

        config = TFTConfig(
            num_numeric_features=2,
            num_categorical_features=1,
            categorical_vocab_sizes=[3],
            hidden_size=16,
            lstm_layers=1,
            attention_heads=2,
            encoder_length=12,
            decoder_length=6,
            quantiles=[0.5],
        )
        model = TemporalFusionTransformer(config)
        enc_num = torch.randn(1, 12, 2)
        enc_cat = torch.randint(0, 2, (1, 12, 1))
        dec_num = torch.randn(1, 6, 2)
        dec_cat = torch.randint(0, 2, (1, 6, 1))
        out = model(enc_num, enc_cat, dec_num, dec_cat)
        for key in (
            "quantile_forecasts",
            "attention_weights",
            "encoder_variable_weights",
            "decoder_variable_weights",
        ):
            assert key in out


# ── Schemas ───────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_query_country_uppercased(self) -> None:
        from api.schemas import QueryRequest

        req = QueryRequest(query="test query here", country="de")
        assert req.country == "DE"

    def test_query_invalid_country(self) -> None:
        from api.schemas import QueryRequest

        with pytest.raises(Exception):
            QueryRequest(query="test query here", country="XX")

    def test_forecast_invalid_country(self) -> None:
        from api.schemas import ForecastRequest

        with pytest.raises(Exception):
            ForecastRequest(country="XX")

    def test_forecast_valid(self) -> None:
        from api.schemas import ForecastRequest

        req = ForecastRequest(country="FR", horizon_hours=24)
        assert req.country == "FR"
        assert req.horizon_hours == 24

    def test_ingest_invalid_country(self) -> None:
        from api.schemas import IngestRequest

        with pytest.raises(Exception):
            IngestRequest(country="ZZ")

    def test_ingest_valid(self) -> None:
        from api.schemas import IngestRequest

        req = IngestRequest(country="es", days_back=3)
        assert req.country == "ES"


# ── Data ingest (synthetic path) ──────────────────────────────────────────────


class TestDataIngest:
    def test_synthetic_load_shape(self) -> None:
        from datetime import datetime

        from data.ingest import _synthetic_load

        df = _synthetic_load("DE", datetime(2023, 1, 1), datetime(2023, 1, 8))
        assert "load_mw" in df.columns
        assert "timestamp" in df.columns
        assert len(df) > 0

    def test_synthetic_weather_shape(self) -> None:
        from datetime import datetime

        from data.ingest import _synthetic_weather

        df = _synthetic_weather("FR", datetime(2023, 1, 1), datetime(2023, 1, 8))
        assert "temperature_2m" in df.columns
        assert len(df) > 0

    def test_build_feature_dataset_returns_df(self) -> None:
        # Mock fetch_weather to avoid real HTTP in CI
        from datetime import datetime

        from data.ingest import build_feature_dataset

        df = build_feature_dataset("DE", datetime(2023, 1, 1), datetime(2023, 1, 3))
        assert "load_mw" in df.columns
        assert "temperature_2m" in df.columns
        assert "hour_of_day" in df.columns
        assert "is_weekend" in df.columns
        assert len(df) > 0

    def test_fetch_entso_falls_back_without_key(self) -> None:
        from datetime import datetime

        from data.ingest import fetch_entso_load

        df = fetch_entso_load("DE", datetime(2023, 1, 1), datetime(2023, 1, 3))
        assert len(df) > 0

    def test_fetch_weather_falls_back_without_package(self) -> None:
        from datetime import datetime

        from data.ingest import _synthetic_weather

        df = _synthetic_weather("NL", datetime(2023, 1, 1), datetime(2023, 1, 3))
        assert len(df) > 0


# ── MLflow tracker (mock) ─────────────────────────────────────────────────────


class TestMLflowTracker:
    def test_log_forecast_metrics(self) -> None:
        with patch("mlflow.log_metrics") as mock_log:
            from mlflow_tracking.tracker import log_forecast_metrics

            log_forecast_metrics(1200.0, 1600.0, 0.03, 0.08, 0.07, step=1)
            mock_log.assert_called_once()
            args = mock_log.call_args[0][0]
            assert "mae" in args
            assert "rmse" in args

    def test_log_tft_params(self) -> None:
        with patch("mlflow.log_params") as mock_log:
            from mlflow_tracking.tracker import log_tft_params

            log_tft_params({"hidden_size": 64, "lstm_layers": 2})
            mock_log.assert_called_once()

    def test_compare_runs_no_experiment(self) -> None:
        with patch("mlflow.tracking.MlflowClient") as mock_client:
            mock_client.return_value.get_experiment_by_name.return_value = None
            from mlflow_tracking.tracker import compare_runs

            result = compare_runs("nonexistent_experiment")
            assert result == []


# ── Core logging ──────────────────────────────────────────────────────────────


class TestLogging:
    def test_get_logger_returns_logger(self) -> None:
        from core.logging import get_logger

        logger = get_logger("test.module")
        assert logger is not None

    def test_setup_logging_runs(self) -> None:
        from core.logging import setup_logging

        setup_logging()  # should not raise
