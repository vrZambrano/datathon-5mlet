"""
Schemas Pydantic para validação de entrada e saída da API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any
from datetime import datetime


# ============================================================================
# SCHEMAS DE ENTRADA (REQUEST)
# ============================================================================

class AlunoFeaturesBase(BaseModel):
    """Features base do aluno para predições."""

    ano: int = Field(..., ge=2020, le=2030, description="Ano dos dados")
    inde: float = Field(..., ge=0, le=10, description="Índice do Desenvolvimento Educacional")
    ieg: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Engajamento")
    ida: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Desempenho Acadêmico")
    ips: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicossocial")
    iaa: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Autoavaliação")
    ian: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Adequação ao Nível")
    ipp: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicopedagógico")
    ipv: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Ponto de Virada")


class RiskPredictionRequest(AlunoFeaturesBase):
    """Request para predição de risco de queda."""

    aluno_id: str = Field(..., min_length=1, description="ID do aluno (RA)")

    # Features temporais
    inde_anterior: Optional[float] = Field(None, ge=0, le=10, description="INDE do ano anterior")
    ieg_anterior: Optional[float] = Field(None, ge=0, le=10, description="IEG do ano anterior")
    ida_anterior: Optional[float] = Field(None, ge=0, le=10, description="IDA do ano anterior")

    # Calculadas (se não fornecidas, serão calculadas)
    delta_inde: Optional[float] = None
    delta_ieg: Optional[float] = None
    delta_ida: Optional[float] = None

    anos_no_programa: Optional[int] = Field(None, ge=1, description="Anos no programa")
    pedra: Optional[str] = Field(None, description="Pedra atual (Quartzo, Ágata, Ametista, Topázio)")

    @field_validator("pedra")
    @classmethod
    def validate_pedra(cls, v):
        if v is not None:
            validas = ["Quartzo", "Ágata", "Ametista", "Topázio"]
            if v not in validas:
                raise ValueError(f"Pedra deve ser uma de: {validas}")
        return v


class ClusterPredictionRequest(AlunoFeaturesBase):
    """Request para predição de cluster."""

    aluno_id: str = Field(..., min_length=1, description="ID do aluno (RA)")


class ReportGenerationRequest(BaseModel):
    """Request para geração de relatório LLM."""

    aluno_id: str = Field(..., min_length=1, description="ID do aluno (RA)")
    nome: str = Field(..., min_length=1, description="Nome do aluno")
    idade: Optional[int] = Field(None, ge=5, le=100, description="Idade do aluno")
    pedra: str = Field(..., description="Pedra atual")
    inde: float = Field(..., ge=0, le=10, description="INDE atual")

    # Informações adicionais
    cluster_nome: Optional[str] = Field(None, description="Nome do cluster")
    risco_percentual: Optional[float] = Field(None, ge=0, le=100, description="Risco de queda (%)")

    # Histórico
    anos_no_programa: Optional[int] = Field(None, ge=1, description="Anos no programa")
    tendencia_inde: Optional[str] = Field(None, description="Tendência do INDE (crescendo/estável/decrescendo)")

    # Feedback (se disponível)
    feedback_texto: Optional[str] = Field(None, description="Feedback qualitativo do aluno")


# ============================================================================
# SCHEMAS DE SAÍDA (RESPONSE)
# ============================================================================

class RiskPredictionResponse(BaseModel):
    """Response para predição de risco."""

    aluno_id: str
    risco_probabilidade: float = Field(..., ge=0, le=1, description="Probabilidade de queda (0-1)")
    risco_classe: str = Field(..., description="Classe de risco (BAIXO, MEDIO, ALTO)")
    vai_cair: bool = Field(..., description="Se o aluno vai cair de pedra")

    # Features mais importantes
    features_importantes: Optional[Dict[str, float]] = Field(None, description="Importância das features")

    timestamp: datetime = Field(default_factory=datetime.now)


class ClusterPredictionResponse(BaseModel):
    """Response para predição de cluster."""

    aluno_id: str
    cluster_id: int
    cluster_nome: str
    cluster_descricao: Optional[str] = None

    timestamp: datetime = Field(default_factory=datetime.now)


class ReportGenerationResponse(BaseModel):
    """Response para geração de relatório."""

    aluno_id: str
    relatorio: str
    modelo_llm: str
    tokens_usados: Optional[int] = None

    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Response para health check."""

    status: str
    api_version: str = "1.0.0"
    models_loaded: bool = False
    llm_configured: bool = False

    services: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Response para erros."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# SCHEMAS DE DASHBOARD
# ============================================================================

class DashboardMetrics(BaseModel):
    """Métricas gerais para o dashboard."""

    total_alunos: int
    distribuicao_pedras: Dict[str, int]
    inde_medio: float
    inde_por_ano: Dict[str, float]

    alunos_por_cluster: Optional[Dict[str, int]] = None
    alunos_risco_alto: Optional[int] = None


class AlunoHistoricoResponse(BaseModel):
    """Histórico de um aluno."""

    aluno_id: str
    nome: Optional[str] = None
    historico: List[Dict[str, Any]]

    # Predições atuais
    risco_atual: Optional[RiskPredictionResponse] = None
    cluster_atual: Optional[ClusterPredictionResponse] = None
