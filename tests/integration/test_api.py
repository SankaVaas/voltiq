"""tests/integration/test_api.py — FastAPI endpoint integration tests."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert data["status"] in ("ok", "degraded", "error")


class TestForecastEndpoint:
    def test_forecast_default(self):
        response = client.post("/api/v1/forecast", json={"country": "DE"})
        assert response.status_code == 200
        data = response.json()
        assert data["country"] == "DE"
        assert len(data["p50"]) == 48
        assert len(data["timestamps"]) == 48

    def test_forecast_custom_horizon(self):
        response = client.post("/api/v1/forecast", json={"country": "FR", "horizon_hours": 24})
        assert response.status_code == 200
        data = response.json()
        assert len(data["p50"]) == 24

    def test_forecast_invalid_country(self):
        response = client.post("/api/v1/forecast", json={"country": "XX"})
        assert response.status_code == 422


class TestAnomalyEndpoint:
    def test_anomaly_scan(self):
        response = client.get("/api/v1/anomalies?country=DE")
        assert response.status_code == 200
        data = response.json()
        assert "anomaly_count" in data
        assert "severity" in data
        assert data["severity"] in ("low", "medium", "high")
        assert 0.0 <= data["anomaly_rate"] <= 1.0


class TestIngestEndpoint:
    def test_ingest_returns_status(self):
        response = client.post("/api/v1/ingest", json={"country": "DE", "days_back": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "error")
        assert data["country"] == "DE"


class TestMetrics:
    def test_prometheus_metrics_exposed(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"voltiq_http_requests_total" in response.content
