"""
API FastAPI principal para o Assistente Pedagógico Passos Mágicos.

Esta API fornece endpoints para:
- Predição de risco de queda de pedra
- Clusterização de perfis de alunos
- Geração de relatórios com LLM
- Health check e métricas
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.core.config import get_settings
from app.core.logger import setup_logging

# Import routers
from app.routes import health, predict, cluster, enrich, web

# Import exceptions (se houver)

# Configurações
settings = get_settings()

# Configura logging
setup_logging()

# Cria aplicação FastAPI
app = FastAPI(
    title="Passos Mágicos - Assistente Pedagógico API",
    description="API de Machine Learning para predição de risco e clusterização de alunos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# EVENTOS DE STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Executado na inicialização da API."""
    logger.info("=" * 50)
    logger.info("INICIANDO API PASSOS MÁGICOS")
    logger.info(f"Ambiente: {settings.environment}")
    logger.info(f"API Host: {settings.api_host}:{settings.api_port}")
    logger.info("=" * 50)

    # Tenta carregar modelos (lazy loading, apenas verifica existência)
    from pathlib import Path

    model_files = [
        settings.classifier_model,
        settings.clustering_model,
        settings.scaler
    ]

    models_loaded = all(Path(m).exists() for m in model_files)

    if models_loaded:
        logger.info("✓ Modelos encontrados")
    else:
        logger.warning("✗ Modelos não encontrados. A API funcionará mas as predições falharão.")

    # Verifica LLM
    llm_configured = bool(settings.openrouter_api_key)
    if llm_configured:
        logger.info("✓ LLM configurado")
    else:
        logger.warning("✗ LLM não configurado (OPENROUTER_API_KEY não definida)")


@app.on_event("shutdown")
async def shutdown_event():
    """Executado no desligamento da API."""
    logger.info("Desligando API...")


# ============================================================================
# ROTAS PRINCIPAIS
# ============================================================================

# Root
@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "nome": "Passos Mágicos - Assistente Pedagógico",
        "versao": "1.0.0",
        "status": "operacional",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict_risk": "/predict/risk",
            "predict_cluster": "/predict/cluster",
            "enrich_report": "/enrich/report"
        }
    }


# Static files for Web UI
app.mount("/static", StaticFiles(directory="frontend_web/static"), name="static")

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(cluster.router, prefix="/predict", tags=["Clustering"])
app.include_router(enrich.router, prefix="/enrich", tags=["LLM Enrichment"])
app.include_router(web.router, prefix="/ui", tags=["Web UI"])


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handler para erros de valor."""
    logger.warning(f"ValueError: {exc}")
    from app.models.schemas import ErrorResponse
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Valor inválido", "detail": str(exc)}
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handler para arquivos não encontrados."""
    logger.error(f"FileNotFoundError: {exc}")
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Recurso não encontrado", "detail": "Modelo ou arquivo não encontrado"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler genérico para exceções não tratadas."""
    logger.error(f"Exception não tratada: {type(exc).__name__}: {exc}")
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Erro interno", "detail": str(exc) if settings.is_development else None}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )
