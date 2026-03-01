"""
Testes para app/routes/predict.py
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestPredictRiskEndpoint:
    """Testes para POST /predict/risk."""

    @pytest.mark.asyncio
    async def test_predict_risk_with_valid_data(self, api_client):
        """Testa predição com dados válidos."""
        payload = {
            "aluno_id": "RA-001",
            "ano": 2024,
            "inde": 7.5,
            "ieg": 6.0,
            "ida": 7.0,
            "ips": 5.5,
            "iaa": 6.5,
            "delta_inde": 0.5,
            "delta_ieg": 0.3,
            "delta_ida": 0.2,
            "anos_no_programa": 2,
            "tendencia_inde": 0.5,
            "pedras_mudadas_total": 1.0,
        }

        async with api_client as client:
            response = await client.post("/predict/risk", json=payload)

        # 200 se modelo existe, 503 se não existe
        if response.status_code == 200:
            data = response.json()
            assert "risco_probabilidade" in data
            assert "risco_classe" in data
            assert "vai_cair" in data
            assert data["risco_classe"] in ["BAIXO", "MEDIO", "ALTO"]
            assert 0 <= data["risco_probabilidade"] <= 1
        else:
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_predict_risk_missing_required_fields(self, api_client):
        """Testa predição sem campos obrigatórios."""
        payload = {"aluno_id": "RA-001"}  # Missing ano, inde

        async with api_client as client:
            response = await client.post("/predict/risk", json=payload)
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_predict_risk_invalid_inde_range(self, api_client):
        """Testa predição com INDE fora do range."""
        payload = {
            "aluno_id": "RA-001",
            "ano": 2024,
            "inde": 15.0,  # Max é 10
        }

        async with api_client as client:
            response = await client.post("/predict/risk", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_risk_minimal_data(self, api_client):
        """Testa predição com dados mínimos."""
        payload = {
            "aluno_id": "RA-001",
            "ano": 2024,
            "inde": 5.0,
        }

        async with api_client as client:
            response = await client.post("/predict/risk", json=payload)
        # Should work (optionals default to None/0) or 503 if no model
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_predict_risk_empty_aluno_id(self, api_client):
        """Testa predição com aluno_id vazio."""
        payload = {
            "aluno_id": "",
            "ano": 2024,
            "inde": 5.0,
        }

        async with api_client as client:
            response = await client.post("/predict/risk", json=payload)
        assert response.status_code == 422


class TestCalculateDeltasFromRequest:
    """Testes para calculate_deltas_from_request."""

    def test_calculates_delta_inde(self):
        """Testa cálculo de delta INDE."""
        from app.routes.predict import calculate_deltas_from_request
        from app.models.schemas import RiskPredictionRequest

        request = RiskPredictionRequest(
            aluno_id="RA-001", ano=2024, inde=7.5,
            inde_anterior=5.0, delta_inde=None,
        )

        deltas = calculate_deltas_from_request(request)
        assert "delta_INDE" in deltas
        assert deltas["delta_INDE"] == pytest.approx(2.5)

    def test_no_delta_when_provided(self):
        """Testa que não recalcula quando delta fornecido."""
        from app.routes.predict import calculate_deltas_from_request
        from app.models.schemas import RiskPredictionRequest

        request = RiskPredictionRequest(
            aluno_id="RA-001", ano=2024, inde=7.5,
            inde_anterior=5.0, delta_inde=1.0,
        )

        deltas = calculate_deltas_from_request(request)
        # Quando delta_inde é fornecido, não recalcula
        assert "delta_INDE" not in deltas
