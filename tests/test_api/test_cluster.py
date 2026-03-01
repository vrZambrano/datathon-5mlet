"""
Testes para app/routes/cluster.py
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPredictClusterEndpoint:
    """Testes para POST /predict/cluster."""

    @pytest.mark.asyncio
    async def test_predict_cluster_valid_data(self, api_client):
        """Testa predição de cluster com dados válidos."""
        payload = {
            "aluno_id": "RA-001",
            "ano": 2024,
            "inde": 7.5,
            "ieg": 6.0,
            "ida": 7.0,
            "ips": 5.5,
            "iaa": 6.5,
        }

        async with api_client as client:
            response = await client.post("/predict/cluster", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "cluster_id" in data
            assert "cluster_nome" in data
            assert "aluno_id" in data
            assert data["aluno_id"] == "RA-001"
            assert isinstance(data["cluster_id"], int)
        else:
            assert response.status_code == 503  # Model not available

    @pytest.mark.asyncio
    async def test_predict_cluster_missing_fields(self, api_client):
        """Testa predição sem campos obrigatórios."""
        payload = {"aluno_id": "RA-001"}

        async with api_client as client:
            response = await client.post("/predict/cluster", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_cluster_minimal(self, api_client):
        """Testa predição com dados mínimos."""
        payload = {
            "aluno_id": "RA-001",
            "ano": 2024,
            "inde": 5.0,
        }

        async with api_client as client:
            response = await client.post("/predict/cluster", json=payload)
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_predict_cluster_invalid_inde(self, api_client):
        """Testa com INDE inválido."""
        payload = {
            "aluno_id": "RA-001",
            "ano": 2024,
            "inde": -1.0,  # Negative
        }

        async with api_client as client:
            response = await client.post("/predict/cluster", json=payload)
        assert response.status_code == 422
