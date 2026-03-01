"""
Testes para app/main.py - rotas raiz e exception handlers.
"""

import pytest


class TestRootEndpoint:
    """Testes para GET /."""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, api_client):
        """Testa raiz retorna 200."""
        async with api_client as client:
            response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "nome" in data
        assert "versao" in data
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_root_has_correct_version(self, api_client):
        """Testa que versão é 1.0.0."""
        async with api_client as client:
            response = await client.get("/")
        data = response.json()
        assert data["versao"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_root_endpoints_listed(self, api_client):
        """Testa que endpoints principais estão listados."""
        async with api_client as client:
            response = await client.get("/")
        data = response.json()
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "predict_risk" in endpoints
        assert "predict_cluster" in endpoints
        assert "enrich_report" in endpoints


class TestExceptionHandlers:
    """Testes para exception handlers."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, api_client):
        """Testa envio de JSON inválido."""
        async with api_client as client:
            response = await client.post(
                "/predict/risk",
                content="not json",
                headers={"Content-Type": "application/json"}
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_route(self, api_client):
        """Testa rota que não existe."""
        async with api_client as client:
            response = await client.get("/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_wrong_method(self, api_client):
        """Testa método HTTP errado."""
        async with api_client as client:
            response = await client.get("/predict/risk")
        assert response.status_code == 405


class TestDocsEndpoints:
    """Testa docs do FastAPI."""

    @pytest.mark.asyncio
    async def test_docs_available(self, api_client):
        """Testa que /docs está acessível."""
        async with api_client as client:
            response = await client.get("/docs")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_spec(self, api_client):
        """Testa que o schema OpenAPI é acessível."""
        async with api_client as client:
            response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
