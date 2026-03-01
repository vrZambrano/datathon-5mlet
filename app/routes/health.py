"""
Rotas de health check e monitoramento.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
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


def _load_processed_data() -> pd.DataFrame:
    """Carrega e processa os CSVs para estatísticas do dashboard."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.loader import load_all_years
    from src.data.preprocessing import harmonize_datasets, normalize_pedra_column
    from src.data.feature_engineering import create_all_temporal_features

    datasets = load_all_years()
    if not datasets:
        return pd.DataFrame()

    df = harmonize_datasets(
        df_2022=datasets.get(2022),
        df_2023=datasets.get(2023),
        df_2024=datasets.get(2024),
    )
    df = normalize_pedra_column(df)
    df = create_all_temporal_features(df)
    return df


@router.get("/stats")
async def get_stats():
    """
    Retorna estatísticas agregadas para o dashboard.

    Inclui: total de alunos, INDE médio, distribuição de pedras,
    distribuição por clusters.
    """
    try:
        df = _load_processed_data()

        if df.empty:
            return {"error": "Nenhum dado disponível"}

        # Dados mais recentes de cada aluno
        df_latest = df.sort_values("ano").groupby("RA").last().reset_index()

        total_alunos = len(df_latest)

        # INDE médio
        inde_medio = None
        if "INDE" in df_latest.columns:
            inde_medio = round(float(df_latest["INDE"].dropna().mean()), 2)

        # Distribuição de pedras
        pedra_dist = {}
        if "PEDRA" in df_latest.columns:
            counts = df_latest["PEDRA"].dropna().value_counts()
            pedra_dist = {str(k): int(v) for k, v in counts.items()}

        # Risco alto - usa o classificador se disponível
        risco_alto = 0
        try:
            classifier_path = Path(settings.classifier_model)
            if classifier_path.exists():
                model = joblib.load(classifier_path)
                feature_cols = [
                    "INDE", "IEG", "IDA", "IPS", "IAA",
                    "delta_INDE", "delta_IEG", "delta_IDA",
                    "anos_no_programa", "tendencia_INDE",
                    "pedras_mudadas_total"
                ]
                available = [c for c in feature_cols if c in df_latest.columns]
                df_pred = df_latest[available].dropna()
                if not df_pred.empty:
                    # Preenche colunas faltantes com 0
                    for c in feature_cols:
                        if c not in df_pred.columns:
                            df_pred[c] = 0.0
                    df_pred = df_pred[feature_cols]
                    probs = model.predict_proba(df_pred)[:, 1]
                    risco_alto = int((probs >= settings.risk_threshold).sum())
        except Exception as e:
            logger.warning(f"Erro ao calcular risco: {e}")

        # Distribuição de clusters
        cluster_dist = {}
        try:
            clustering_path = Path(settings.clustering_model)
            scaler_path = Path(settings.scaler)
            labels_path = Path("models/cluster_labels.json")

            if clustering_path.exists() and scaler_path.exists():
                km = joblib.load(clustering_path)
                scaler = joblib.load(scaler_path)

                cluster_feats = ["INDE", "IEG", "IDA", "IPS", "IAA"]
                available_cf = [c for c in cluster_feats if c in df_latest.columns]
                df_cl = df_latest[available_cf].dropna()

                if not df_cl.empty:
                    X_scaled = scaler.transform(df_cl[cluster_feats])
                    labels = km.predict(X_scaled)

                    # Carrega nomes dos clusters
                    cluster_names = {}
                    if labels_path.exists():
                        with open(labels_path) as f:
                            ldata = json.load(f)
                        cluster_names = ldata.get("cluster_names", {})

                    for cid in sorted(set(labels)):
                        name = cluster_names.get(str(cid), f"Cluster {cid}")
                        cluster_dist[name] = int((labels == cid).sum())
        except Exception as e:
            logger.warning(f"Erro ao calcular clusters: {e}")

        # Evolução INDE por ano
        inde_por_ano = {}
        if "INDE" in df.columns and "ano" in df.columns:
            inde_por_ano = {
                str(int(ano)): round(float(grp["INDE"].dropna().mean()), 2)
                for ano, grp in df.groupby("ano") if not grp["INDE"].dropna().empty
            }

        return {
            "total_alunos": total_alunos,
            "inde_medio": inde_medio,
            "risco_alto": risco_alto,
            "n_clusters": 4,
            "pedra_distribuicao": pedra_dist,
            "cluster_distribuicao": cluster_dist,
            "inde_por_ano": inde_por_ano,
            "anos_disponiveis": sorted(df["ano"].unique().tolist()) if "ano" in df.columns else [],
        }
    except Exception as e:
        logger.error(f"Erro ao gerar stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students")
async def get_students(ano: int = None):
    """
    Retorna lista de alunos com seus indicadores para seleção no frontend.

    Parâmetros:
        ano: Filtrar por ano específico (opcional, padrão = ano mais recente)
    """
    try:
        df = _load_processed_data()
        if df.empty:
            return {"students": []}

        # Se ano não especificado, usa todos os anos
        if ano:
            df_year = df[df["ano"] == ano]
        else:
            df_year = df

        if df_year.empty:
            return {"students": [], "anos_disponiveis": []}

        # Propaga NOME de outros anos para alunos sem nome
        if "NOME" in df.columns:
            name_lookup = df.dropna(subset=["NOME"]).drop_duplicates(subset=["RA"], keep="first")[["RA", "NOME"]]
            if "NOME" in df_year.columns:
                df_year = df_year.drop(columns=["NOME"])
            df_year = df_year.merge(name_lookup, on="RA", how="left")

        # Colunas relevantes para retorno
        cols_map = {
            "RA": "ra", "NOME": "nome", "ano": "ano",
            "INDE": "inde", "IEG": "ieg", "IDA": "ida",
            "IPS": "ips", "IAA": "iaa", "IAN": "ian",
            "IPV": "ipv", "IPP": "ipp",
            "PEDRA": "pedra", "FASE": "fase", "TURMA": "turma",
            "INSTITUICAO DE ENSINO": "instituicao",
            "ANO INGRESSO": "ano_ingresso",
            "tendencia_INDE": "tendencia_inde",
            "pedras_mudadas_total": "pedras_mudadas_total",
            "anos_no_programa": "anos_no_programa",
            "delta_INDE": "delta_inde",
            "delta_IEG": "delta_ieg",
            "delta_IDA": "delta_ida",
        }

        available_cols = {k: v for k, v in cols_map.items() if k in df_year.columns}
        df_out = df_year[list(available_cols.keys())].rename(columns=available_cols)

        # Converte NaN para None para JSON (replace NaN em colunas numéricas)
        for col in df_out.select_dtypes(include=["float64", "float32"]).columns:
            df_out[col] = df_out[col].where(df_out[col].notna(), None)
        # Converte NaN em colunas object
        df_out = df_out.where(df_out.notna(), None)

        # Ordena por nome/RA
        sort_col = "nome" if "nome" in df_out.columns else "ra"
        df_out = df_out.sort_values([sort_col, "ano"])

        anos_disponiveis = sorted(df["ano"].unique().tolist()) if "ano" in df.columns else []

        # Converte para records com NaN -> None
        import math
        records = []
        for _, row in df_out.iterrows():
            rec = {}
            for col, val in row.items():
                if val is None:
                    rec[col] = None
                elif isinstance(val, (float, np.floating)):
                    if math.isnan(val) or math.isinf(val):
                        rec[col] = None
                    else:
                        rec[col] = round(float(val), 2)
                elif isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                else:
                    rec[col] = val
            records.append(rec)

        return {
            "students": records,
            "total": len(records),
            "anos_disponiveis": anos_disponiveis,
        }
    except Exception as e:
        logger.error(f"Erro ao listar alunos: {e}")
        raise HTTPException(status_code=500, detail=str(e))
