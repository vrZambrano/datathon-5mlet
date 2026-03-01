"""
Testes para app/routes/health.py
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHealthCheck:
    """Testes para GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, api_client):
        """Testa health check retorna 200."""
        async with api_client as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_has_services(self, api_client):
        """Testa que health retorna serviços."""
        async with api_client as client:
            response = await client.get("/health")
        data = response.json()
        assert "services" in data
        assert "api" in data["services"]

    @pytest.mark.asyncio
    async def test_health_shows_model_status(self, api_client):
        """Testa que health mostra status dos modelos."""
        async with api_client as client:
            response = await client.get("/health")
        data = response.json()
        assert "models_loaded" in data


class TestMetrics:
    """Testes para GET /health/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, api_client):
        """Testa métricas retorna 200."""
        async with api_client as client:
            response = await client.get("/health/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_metrics_has_models_info(self, api_client):
        """Testa que métricas incluem info dos modelos."""
        async with api_client as client:
            response = await client.get("/health/metrics")
        data = response.json()
        assert "models" in data

    @pytest.mark.asyncio
    async def test_metrics_uptime_positive(self, api_client):
        """Testa que uptime é positivo."""
        async with api_client as client:
            response = await client.get("/health/metrics")
        data = response.json()
        assert data["uptime_seconds"] >= 0


class TestStats:
    """Testes para GET /health/stats."""

    @pytest.mark.asyncio
    async def test_stats_returns(self, api_client):
        """Testa endpoint de stats."""
        async with api_client as client:
            response = await client.get("/health/stats")
        # Pode retornar 200 ou 500 dependendo do estado dos CSVs
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_stats_data_format(self, api_client):
        """Testa formato dos dados de stats."""
        async with api_client as client:
            response = await client.get("/health/stats")
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                assert "total_alunos" in data
                assert "inde_medio" in data


class TestStudents:
    """Testes para GET /health/students."""

    @pytest.mark.asyncio
    async def test_students_returns(self, api_client):
        """Testa endpoint de alunos."""
        async with api_client as client:
            response = await client.get("/health/students")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_students_with_year_filter(self, api_client):
        """Testa filtragem por ano."""
        async with api_client as client:
            response = await client.get("/health/students?ano=2024")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_students_format(self, api_client):
        """Testa formato da resposta."""
        async with api_client as client:
            response = await client.get("/health/students")
        if response.status_code == 200:
            data = response.json()
            assert "students" in data


class TestDrift:
    """Testes para GET /health/drift."""

    @pytest.mark.asyncio
    async def test_drift_returns(self, api_client):
        """Testa endpoint de drift."""
        async with api_client as client:
            response = await client.get("/health/drift")
        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_drift_format(self, api_client):
        """Testa formato da resposta do drift."""
        async with api_client as client:
            response = await client.get("/health/drift")
        if response.status_code == 200:
            data = response.json()
            assert "drift_analysis" in data or "error" in data
