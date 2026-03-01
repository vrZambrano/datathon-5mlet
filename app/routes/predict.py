"""
Rotas de predição de risco de queda.
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

from app.models.schemas import RiskPredictionRequest, RiskPredictionResponse
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


def get_classifier_model():
    """Carrega o modelo de classificação (lazy loading)."""
    model_path = Path(settings.classifier_model)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = joblib.load(model_path)
    return model


def calculate_deltas_from_request(request: RiskPredictionRequest) -> Dict[str, float]:
    """Calcula deltas se não fornecidos."""
    deltas = {}

    if request.delta_inde is None and request.inde_anterior is not None:
        deltas["delta_INDE"] = request.inde - request.inde_anterior

    if request.delta_ieg is None and request.ieg_anterior is not None and request.ieg is not None:
        deltas["delta_IEG"] = request.ieg - request.ieg_anterior

    if request.delta_ida is None and request.ida_anterior is not None and request.ida is not None:
        deltas["delta_IDA"] = request.ida - request.ida_anterior

    return deltas


@router.post("/risk", response_model=RiskPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_risk(request: RiskPredictionRequest):
    """
    Prediz o risco de queda de pedra para um aluno.

    Retorna a probabilidade de queda e a classificação do risco (BAIXO, MEDIO, ALTO).
    """
    try:
        # Carrega modelo
        model = get_classifier_model()

        # Prepara features
        features = {
            "INDE": request.inde,
            "IEG": request.ieg if request.ieg is not None else 0,
            "IDA": request.ida if request.ida is not None else 0,
            "IPS": request.ips if request.ips is not None else 0,
            "IAA": request.iaa if request.iaa is not None else 0,
            "delta_INDE": request.delta_inde if request.delta_inde is not None else 0,
            "delta_IEG": request.delta_ieg if request.delta_ieg is not None else 0,
            "delta_IDA": request.delta_ida if request.delta_ida is not None else 0,
            "anos_no_programa": request.anos_no_programa if request.anos_no_programa is not None else 1,
        }

        # Calcula deltas se necessário
        deltas = calculate_deltas_from_request(request)
        features.update(deltas)

        # Cria DataFrame
        X = pd.DataFrame([features])

        # Predição
        risk_proba = model.predict_proba(X)[0, 1]
        risk_class = int(model.predict(X)[0])

        # Classificação do risco
        if risk_proba >= 0.7:
            risk_level = "ALTO"
        elif risk_proba >= 0.4:
            risk_level = "MEDIO"
        else:
            risk_level = "BAIXO"

        # Feature importance (se disponível)
        features_importantes = None
        if hasattr(model, "feature_importances_"):
            feature_names = list(features.keys())
            importances = dict(zip(feature_names, model.feature_importances_.astype(float)))
            # Top 3
            features_importantes = dict(sorted(importances.items(), key=lambda x: -x[1])[:3])

        logger.info(f"Predição para aluno {request.aluno_id}: risco={risk_level} ({risk_proba:.2%})")

        return RiskPredictionResponse(
            aluno_id=request.aluno_id,
            risco_probabilidade=float(risk_proba),
            risco_classe=risk_level,
            vai_cair=bool(risk_class),
            features_importantes=features_importantes
        )

    except FileNotFoundError as e:
        logger.error(f"Modelo não encontrado: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de predição não disponível. Treine o modelo primeiro."
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )
