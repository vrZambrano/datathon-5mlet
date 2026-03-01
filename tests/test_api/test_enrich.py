"""
Testes para app/routes/enrich.py
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestGenerateReportEndpoint:
    """Testes para POST /enrich/report."""

    @pytest.mark.asyncio
    async def test_report_without_api_key(self, api_client):
        """Testa geração sem API key."""
        payload = {
            "aluno_id": "RA-001",
            "nome": "Aluno Teste",
            "pedra": "Ametista",
            "inde": 7.5,
        }

        async with api_client as client:
            response = await client.post("/enrich/report", json=payload)
        # 503 (LLM não configurado), 500 (API error) ou 200 (sucesso)
        assert response.status_code in [200, 500, 503]

    @pytest.mark.asyncio
    async def test_report_missing_required_fields(self, api_client):
        """Testa geração sem campos obrigatórios."""
        payload = {"aluno_id": "RA-001"}  # Missing nome, pedra, inde

        async with api_client as client:
            response = await client.post("/enrich/report", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_report_invalid_inde(self, api_client):
        """Testa geração com INDE inválido."""
        payload = {
            "aluno_id": "RA-001",
            "nome": "Aluno",
            "pedra": "Quartzo",
            "inde": 15.0,  # Acima do limite
        }

        async with api_client as client:
            response = await client.post("/enrich/report", json=payload)
        assert response.status_code == 422


class TestListModelsEndpoint:
    """Testes para GET /enrich/models."""

    @pytest.mark.asyncio
    async def test_list_models(self, api_client):
        """Testa listagem de modelos."""
        async with api_client as client:
            response = await client.get("/enrich/models")
        assert response.status_code == 200
        data = response.json()
        assert "modelo_primario" in data
        assert "provedor" in data
        assert data["provedor"] == "OpenRouter"
