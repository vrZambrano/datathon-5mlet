"""
Rotas de enriquecimento com LLM (OpenRouter).
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pathlib import Path
from typing import Optional

from app.models.schemas import ReportGenerationRequest, ReportGenerationResponse
from app.core.config import get_settings
from app.services.llm_service import LLMService

router = APIRouter()
settings = get_settings()


@router.post("/report", response_model=ReportGenerationResponse, status_code=status.HTTP_200_OK)
async def generate_report(request: ReportGenerationRequest):
    """
    Gera um relatório personalizado do aluno usando LLM.

    O relatório inclui:
    - Resumo do perfil do aluno
    - Pontos fortes
    - Pontos de atenção
    - Recomendações para professores
    """
    try:
        # Verifica se LLM está configurado
        if not settings.openrouter_api_key or len(settings.openrouter_api_key) < 10:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM não configurado. Configure OPENROUTER_API_KEY."
            )

        # Cria serviço LLM
        llm_service = LLMService(api_key=settings.openrouter_api_key)

        # Prepara dados do aluno
        aluno_data = {
            "nome": request.nome,
            "idade": request.idade or 0,
            "pedra": request.pedra,
            "inde": request.inde,
            "anos_no_programa": request.anos_no_programa or 1,
            "tendencia_inde": request.tendencia_inde or "estável",
            "cluster_nome": request.cluster_nome or "Não identificado",
            "risco_percentual": int(request.risco_percentual or 0),
            "feedback_texto": request.feedback_texto or "Não disponível"
        }

        # Gera relatório
        logger.info(f"Gerando relatório para aluno {request.aluno_id}")
        relatorio = await llm_service.generate_student_report(aluno_data)

        logger.info(f"Relatório gerado com sucesso: {len(relatorio)} caracteres")

        return ReportGenerationResponse(
            aluno_id=request.aluno_id,
            relatorio=relatorio,
            modelo_llm=settings.openrouter_model
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar relatório: {str(e)}"
        )


@router.get("/models")
async def list_available_models():
    """
    Lista os modelos LLM disponíveis via OpenRouter.
    """
    return {
        "modelo_primario": settings.openrouter_model,
        "modelo_fallback": settings.openrouter_fallback_model,
        "provedor": "OpenRouter",
        "status": "configured" if settings.openrouter_api_key else "not_configured"
    }
