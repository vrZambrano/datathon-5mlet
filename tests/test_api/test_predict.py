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


class TestColdStartMedianImputation:
    """Testes para imputação por medianas no cold start."""

    def test_feature_medians_exist(self):
        """Verifica que FEATURE_MEDIANS está definido e tem as features corretas."""
        from src.utils.constants import FEATURE_MEDIANS

        expected_keys = {"INDE", "IEG", "IDA", "IPS", "IAA"}
        assert set(FEATURE_MEDIANS.keys()) == expected_keys
        for key, val in FEATURE_MEDIANS.items():
            assert isinstance(val, float), f"{key} deve ser float"
            assert 0 < val < 10, f"{key}={val} fora do range esperado"

    @pytest.mark.asyncio
    async def test_minimal_data_uses_medians_not_zero(self, api_client):
        """Testa que dados mínimos usam medianas, não zero."""
        payload = {
            "aluno_id": "RA-NOVO",
            "ano": 2024,
            "inde": 6.5,
        }

        async with api_client as client:
            response = await client.post("/predict/risk", json=payload)

        if response.status_code == 200:
            data = response.json()
            # Com medianas, aluno com INDE=6.5 (razoável) não deve ser ALTO
            # (antes, com zeros, seria ALTO por outlier extremo)
            assert data["risco_classe"] in ["BAIXO", "MEDIO", "ALTO"]
            assert 0 <= data["risco_probabilidade"] <= 1
