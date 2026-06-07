"""tests/unit/test_core.py — unit tests for config, anomaly detector, and schemas."""

import numpy as np
import pytest
import torch

from core.config import Settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.app_name == "voltiq"
        assert s.forecast_horizon == 48
        assert s.forecast_lookback == 168
        assert s.anomaly_threshold == 0.95

    def test_qdrant_url(self):
        s = Settings()
        assert s.qdrant_url == f"http://{s.qdrant_host}:{s.qdrant_port}"

    def test_log_level_validation(self):
        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"

        with pytest.raises(Exception):
            Settings(log_level="INVALID")


class TestLSTMAutoencoder:
    def test_forward_pass(self):
        from anomaly.detector import LSTMAutoencoder
        model = LSTMAutoencoder(window_size=24, hidden_size=32, latent_dim=8, num_layers=1)
        x = torch.randn(4, 24, 1)  # (batch, seq, features)
        recon = model(x)
        assert recon.shape == x.shape

    def test_reconstruction_error_shape(self):
        from anomaly.detector import LSTMAutoencoder
        model = LSTMAutoencoder(window_size=24, hidden_size=32, latent_dim=8, num_layers=1)
        x = torch.randn(8, 24, 1)
        err = model.reconstruction_error(x)
        assert err.shape == (8,)
        assert (err >= 0).all()


class TestAnomalyDetector:
    def test_train_and_detect(self):
        from anomaly.detector import AnomalyDetector

        rng = np.random.default_rng(0)
        normal_series = rng.normal(50_000, 1_000, 200)

        detector = AnomalyDetector(window_size=24, threshold_percentile=90.0)
        losses = detector.train(normal_series, epochs=3, batch_size=16)

        assert len(losses) == 3
        assert all(l >= 0 for l in losses)
        assert detector.threshold is not None

        # Inject spike
        test_series = rng.normal(50_000, 1_000, 100)
        test_series[50:54] *= 2.0

        result = detector.detect(test_series)
        assert "anomaly_indices" in result
        assert "anomaly_rate" in result
        assert 0.0 <= result["anomaly_rate"] <= 1.0

    def test_save_load(self, tmp_path):
        from anomaly.detector import AnomalyDetector

        rng = np.random.default_rng(1)
        series = rng.normal(0, 1, 100)

        det = AnomalyDetector(window_size=10)
        det.train(series, epochs=2)

        path = tmp_path / "model.pt"
        det.save(path)

        det2 = AnomalyDetector(window_size=10, model_path=path)
        assert abs(det2.threshold - det.threshold) < 1e-6


class TestTFT:
    def test_forward_pass(self):
        from forecasting.models.tft import TemporalFusionTransformer, TFTConfig

        config = TFTConfig(
            num_numeric_features=4,
            num_categorical_features=2,
            hidden_size=32,
            lstm_layers=1,
            attention_heads=2,
            encoder_length=24,
            decoder_length=12,
        )
        model = TemporalFusionTransformer(config)

        B = 2
        enc_num = torch.randn(B, 24, 4)
        enc_cat = torch.randint(0, 2, (B, 24, 2))
        dec_num = torch.randn(B, 12, 4)
        dec_cat = torch.randint(0, 2, (B, 12, 2))

        out = model(enc_num, enc_cat, dec_num, dec_cat)

        assert out["quantile_forecasts"].shape == (B, 12, 3)
        assert out["attention_weights"].shape[0] == B


class TestAPISchemas:
    def test_query_request_valid(self):
        from api.schemas import QueryRequest
        req = QueryRequest(query="What is the forecast for Germany?", country="de")
        assert req.country == "DE"

    def test_query_request_invalid_country(self):
        from api.schemas import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="test query here", country="XX")

    def test_query_request_too_short(self):
        from api.schemas import QueryRequest
        with pytest.raises(Exception):
            QueryRequest(query="hi")