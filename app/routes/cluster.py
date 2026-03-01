"""
Rotas de predição de cluster/perfil.
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger
import joblib
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional

from app.models.schemas import ClusterPredictionRequest, ClusterPredictionResponse
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


def get_clustering_model():
    """Carrega o modelo de clusterização (lazy loading)."""
    model_path = Path(settings.clustering_model)
    scaler_path = Path(settings.scaler)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def get_cluster_labels() -> Dict:
    """Carrega os rótulos dos clusters."""
    labels_path = "models/cluster_labels.json"

    if not Path(labels_path).exists():
        return {}

    with open(labels_path, "r") as f:
        return json.load(f)


@router.post("/cluster", response_model=ClusterPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_cluster(request: ClusterPredictionRequest):
    """
    Prediz o cluster/perfil de um aluno.

    Retorna o ID do cluster e seu nome descritivo.
    """
    try:
        # Carrega modelo e scaler
        model, scaler = get_clustering_model()

        # Carrega labels dos clusters
        labels_data = get_cluster_labels()

        # Prepara features
        features = {
            "INDE": request.inde,
            "IEG": request.ieg if request.ieg is not None else 0,
            "IDA": request.ida if request.ida is not None else 0,
            "IPS": request.ips if request.ips is not None else 0,
            "IAA": request.iaa if request.iaa is not None else 0,
        }

        # Cria DataFrame
        X = pd.DataFrame([features])

        # Escala
        X_scaled = scaler.transform(X)

        # Predição
        cluster_id = int(model.predict(X_scaled)[0])

        # Nome do cluster
        cluster_names = labels_data.get("cluster_names", {})
        cluster_nome = cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")

        # Descrição do cluster
        cluster_info = labels_data.get("cluster_info", {}).get(str(cluster_id), {})
        descricao = f"Alunos com perfil {cluster_nome}"

        logger.info(f"Predição para aluno {request.aluno_id}: cluster={cluster_nome}")

        return ClusterPredictionResponse(
            aluno_id=request.aluno_id,
            cluster_id=cluster_id,
            cluster_nome=cluster_nome,
            cluster_descricao=descricao
        )

    except FileNotFoundError as e:
        logger.error(f"Modelo não encontrado: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de clusterização não disponível. Treine o modelo primeiro."
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )
