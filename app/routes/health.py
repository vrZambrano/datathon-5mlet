"""
Rotas de health check e monitoramento.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from pathlib import Path
from app.core.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter()

settings = get_settings()


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check da API.

    Verifica status dos serviços e modelos.
    """
    # Verifica modelos
    model_files = [
        settings.classifier_model,
        settings.clustering_model,
        settings.scaler
    ]
    models_loaded = all(Path(m).exists() for m in model_files)

    # Verifica LLM
    llm_configured = bool(settings.openrouter_api_key and len(settings.openrouter_api_key) > 10)

    # Status dos serviços
    services = {
        "api": "ok",
        "classifier": "loaded" if Path(settings.classifier_model).exists() else "not_found",
        "clustering": "loaded" if Path(settings.clustering_model).exists() else "not_found",
        "llm": "configured" if llm_configured else "not_configured"
    }

    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        llm_configured=llm_configured,
        services=services
    )


@router.get("/metrics")
async def get_metrics():
    """
    Retorna métricas básicas da API.

    Em produção, isso pode incluir métricas de Prometheus.
    """
    return {
        "status": "ok",
        "uptime": "implementar",
        "requests_total": "implementar",
        "errors_total": "implementar"
    }
