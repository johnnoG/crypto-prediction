"""
Tests for Health Check Endpoints

Tests all health monitoring endpoints:
- Basic health check
- Database health
- Cache health
- Detailed health check
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime


class TestHealthEndpoints:
    """Test cases for health check endpoints."""

    @pytest.mark.integration
    def test_basic_health_check(self, client: TestClient):
        """Test GET /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data

        # Verify timestamp format
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert isinstance(timestamp, datetime)

    @pytest.mark.integration
    @pytest.mark.db
    def test_database_health_check(self, client: TestClient):
        """Test GET /health/db endpoint."""
        response = client.get("/health/db")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert "latency_ms" in data
        assert "timestamp" in data
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0

    @pytest.mark.integration
    @pytest.mark.cache
    def test_cache_health_check(self, client: TestClient):
        """Test GET /health/cache endpoint."""
        response = client.get("/health/cache")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["connected"] is True
        assert "timestamp" in data

    @pytest.mark.integration
    def test_detailed_health_check(self, client: TestClient):
        """Test GET /health/detailed endpoint."""
        response = client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        # Overall status
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

        # Service info
        assert "service" in data
        assert "name" in data["service"]
        assert "version" in data["service"]
        assert "environment" in data["service"]

        # Database info
        assert "database" in data
        assert "status" in data["database"]
        assert "connected" in data["database"]

        # Cache info
        assert "cache" in data
        assert "status" in data["cache"]
        assert "connected" in data["cache"]

        # Timestamp
        assert "timestamp" in data


class TestHealthCheckResponses:
    """Test health check response formats and data."""

    @pytest.mark.integration
    def test_health_check_response_structure(self, client: TestClient):
        """Test that health check response has correct structure."""
        response = client.get("/health")
        data = response.json()

        # Required fields
        required_fields = ["status", "service", "version", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    @pytest.mark.integration
    def test_database_health_latency_measurement(self, client: TestClient):
        """Test that database latency is measured."""
        response = client.get("/health/db")
        data = response.json()

        assert "latency_ms" in data
        latency = data["latency_ms"]

        # Latency should be positive and reasonable (< 1000ms for local)
        assert latency > 0
        assert latency < 1000

    @pytest.mark.integration
    def test_cache_health_metrics(self, client: TestClient):
        """Test that cache health includes metrics."""
        response = client.get("/health/cache")
        data = response.json()

        # Should have status
        assert "status" in data
        assert "connected" in data

    @pytest.mark.integration
    def test_detailed_health_aggregation(self, client: TestClient):
        """Test that detailed health properly aggregates status."""
        response = client.get("/health/detailed")
        data = response.json()

        # If both DB and cache are healthy, overall should be healthy
        if (data["database"]["status"] == "healthy" and
            data["cache"]["status"] == "healthy"):
            assert data["status"] == "healthy"

        # If DB is unhealthy, overall should be unhealthy
        # (This would require mocking DB failure)


class TestHealthCheckEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.integration
    def test_health_check_always_responds(self, client: TestClient):
        """Test that basic health check always responds even if deps fail."""
        # Basic health should work without dependencies
        response = client.get("/health")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_multiple_health_checks(self, client: TestClient):
        """Test multiple consecutive health checks."""
        # Make multiple requests
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    @pytest.mark.integration
    def test_health_check_timestamps_increase(self, client: TestClient):
        """Test that timestamps increase over time."""
        response1 = client.get("/health")
        timestamp1 = datetime.fromisoformat(response1.json()["timestamp"])

        import time
        time.sleep(0.1)  # Small delay

        response2 = client.get("/health")
        timestamp2 = datetime.fromisoformat(response2.json()["timestamp"])

        assert timestamp2 > timestamp1


class TestHealthCheckPerformance:
    """Test health check performance characteristics."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_health_check_performance(self, client: TestClient):
        """Test that health checks respond quickly."""
        import time

        start = time.time()
        response = client.get("/health")
        duration = time.time() - start

        assert response.status_code == 200
        # Health check should be very fast (< 100ms)
        assert duration < 0.1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_health_performance(self, client: TestClient):
        """Test database health check performance."""
        import time

        start = time.time()
        response = client.get("/health/db")
        duration = time.time() - start

        assert response.status_code == 200
        # Should complete quickly (< 500ms)
        assert duration < 0.5

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cache_health_performance(self, client: TestClient):
        """Test cache health check performance."""
        import time

        start = time.time()
        response = client.get("/health/cache")
        duration = time.time() - start

        assert response.status_code == 200
        # Cache health should be very fast (< 100ms)
        assert duration < 0.1
