"""tests/integration/test_api.py — FastAPI endpoint integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert data["status"] in ("ok", "degraded", "error")

    def test_health_version_matches(self) -> None:
        from core.config import settings
        data = client.get("/health").json()
        assert data["version"] == settings.app_version


# ── Forecast ─────────────────────────────────────────────────────────────────

class TestForecastEndpoint:
    def test_forecast_default(self) -> None:
        response = client.post("/api/v1/forecast", json={"country": "DE"})
        assert response.status_code == 200
        data = response.json()
        assert data["country"] == "DE"
        assert len(data["p50"]) == 48
        assert len(data["timestamps"]) == 48

    def test_forecast_custom_horizon(self) -> None:
        response = client.post("/api/v1/forecast", json={"country": "FR", "horizon_hours": 24})
        assert response.status_code == 200
        assert len(response.json()["p50"]) == 24

    def test_forecast_invalid_country(self) -> None:
        response = client.post("/api/v1/forecast", json={"country": "XX"})
        assert response.status_code == 422

    def test_forecast_all_countries(self) -> None:
        for country in ("DE", "FR", "ES", "NL", "PL"):
            resp = client.post("/api/v1/forecast", json={"country": country})
            assert resp.status_code == 200
            assert resp.json()["country"] == country

    def test_forecast_p10_less_than_p90(self) -> None:
        data = client.post("/api/v1/forecast", json={"country": "DE"}).json()
        for p10, p90 in zip(data["p10"], data["p90"], strict=False):
            assert p10 < p90

    def test_forecast_has_peak(self) -> None:
        data = client.post("/api/v1/forecast", json={"country": "DE"}).json()
        assert data["peak_forecast_mw"] > 0


# ── Anomaly ───────────────────────────────────────────────────────────────────

class TestAnomalyEndpoint:
    def test_anomaly_scan_returns_200(self) -> None:
        response = client.get("/api/v1/anomalies?country=DE")
        assert response.status_code == 200

    def test_anomaly_schema(self) -> None:
        data = client.get("/api/v1/anomalies?country=DE").json()
        assert "anomaly_count" in data
        assert "severity" in data
        assert "anomaly_rate" in data
        assert "threshold" in data
        assert data["severity"] in ("low", "medium", "high")
        assert 0.0 <= data["anomaly_rate"] <= 1.0

    def test_anomaly_indices_is_list(self) -> None:
        data = client.get("/api/v1/anomalies?country=FR").json()
        assert isinstance(data["anomaly_indices"], list)


# ── Ingest ────────────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_ingest_returns_status_ok(self) -> None:
        response = client.post("/api/v1/ingest", json={"country": "DE", "days_back": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "error")
        assert data["country"] == "DE"

    def test_ingest_invalid_country(self) -> None:
        response = client.post("/api/v1/ingest", json={"country": "XX", "days_back": 1})
        assert response.status_code == 422

    def test_ingest_ok_has_rows(self) -> None:
        response = client.post("/api/v1/ingest", json={"country": "DE", "days_back": 1})
        data = response.json()
        if data["status"] == "ok":
            assert data["rows_ingested"] > 0


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_prometheus_metrics_exposed(self) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"voltiq_http_requests_total" in response.content

    def test_metrics_after_requests(self) -> None:
        client.get("/health")
        response = client.get("/metrics")
        assert b"voltiq_http_requests_total" in response.content


# ── Schemas (via API) ─────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_query_too_short_rejected(self) -> None:
        # query min_length=3
        response = client.post("/api/v1/forecast", json={"country": "DE", "horizon_hours": 0})
        assert response.status_code == 422

    def test_horizon_max_rejected(self) -> None:
        response = client.post("/api/v1/forecast", json={"country": "DE", "horizon_hours": 999})
        assert response.status_code == 422

    def test_ingest_days_back_max(self) -> None:
        response = client.post("/api/v1/ingest", json={"country": "DE", "days_back": 9999})
        assert response.status_code == 422
