"""
Monitoramento de Data Drift usando Evidently AI.

Este módulo compara distribuições de dados de referência (treino)
com dados atuais (produção) para identificar drift.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)


def create_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Report:
    """
    Gera um relatório de data drift comparando dados de referência com atuais.

    Args:
        reference_data: DataFrame de referência (dados de treino)
        current_data: DataFrame com dados atuais (produção)
        feature_cols: Lista de features para monitorar (opcional, usa todas numéricas)

    Returns:
        Report do Evidently AI
    """
    if feature_cols:
        ref = reference_data[feature_cols].copy()
        cur = current_data[feature_cols].copy()
    else:
        ref = reference_data.select_dtypes(include=[np.number]).copy()
        cur = current_data.select_dtypes(include=[np.number]).copy()

    # Garante que ambas tenham as mesmas colunas
    common_cols = list(set(ref.columns) & set(cur.columns))
    ref = ref[common_cols]
    cur = cur[common_cols]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    report.run(reference_data=ref, current_data=cur)

    logger.info("Relatório de drift gerado com sucesso")
    return report


def check_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    drift_threshold: float = 0.5,
) -> Dict:
    """
    Verifica se há drift significativo nos dados.

    Args:
        reference_data: DataFrame de referência
        current_data: DataFrame atual
        feature_cols: Features para verificar
        drift_threshold: Limiar para considerar drift (proporção de features com drift)

    Returns:
        Dicionário com resultado do drift:
        - dataset_drift: bool (True se drift detectado)
        - drift_share: float (proporção de features com drift)
        - n_drifted_features: int
        - feature_drift: dict com status por feature
    """
    report = create_drift_report(reference_data, current_data, feature_cols)

    result = report.as_dict()

    # Extrai resultados do DatasetDriftMetric
    metrics = result.get("metrics", [])

    dataset_drift = False
    drift_share = 0.0
    n_drifted = 0
    n_features = 0
    feature_drift = {}

    for metric in metrics:
        metric_id = metric.get("metric", "")

        if metric_id == "DatasetDriftMetric":
            metric_result = metric.get("result", {})
            dataset_drift = metric_result.get("dataset_drift", False)
            drift_share = metric_result.get("drift_share", 0.0)
            n_drifted = metric_result.get("number_of_drifted_columns", 0)
            n_features = metric_result.get("number_of_columns", 0)

        elif metric_id == "DataDriftTable":
            metric_result = metric.get("result", {})
            drift_by_columns = metric_result.get("drift_by_columns", {})
            for col_name, col_info in drift_by_columns.items():
                feature_drift[col_name] = {
                    "drift_detected": col_info.get("drift_detected", False),
                    "drift_score": round(col_info.get("drift_score", 0.0), 4),
                    "stattest_name": col_info.get("stattest_name", ""),
                }

    summary = {
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4),
        "n_drifted_features": n_drifted,
        "n_total_features": n_features,
        "drift_threshold": drift_threshold,
        "above_threshold": drift_share > drift_threshold,
        "feature_drift": feature_drift,
    }

    if dataset_drift:
        logger.warning(f"DRIFT DETECTADO: {n_drifted}/{n_features} features ({drift_share:.1%})")
    else:
        logger.info(f"Sem drift significativo: {n_drifted}/{n_features} features ({drift_share:.1%})")

    return summary


def save_drift_report_html(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
    feature_cols: Optional[List[str]] = None,
) -> str:
    """
    Gera e salva um relatório de drift em HTML.

    Args:
        reference_data: DataFrame de referência
        current_data: DataFrame atual
        output_path: Caminho para salvar o HTML
        feature_cols: Features para monitorar

    Returns:
        Caminho do arquivo salvo
    """
    report = create_drift_report(reference_data, current_data, feature_cols)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    report.save_html(output_path)
    logger.info(f"Relatório de drift salvo em: {output_path}")

    return output_path


def compare_year_drift(
    df: pd.DataFrame,
    reference_year: int,
    current_year: int,
    feature_cols: Optional[List[str]] = None,
    ano_col: str = "ano",
) -> Dict:
    """
    Compara drift entre dois anos específicos dos dados.

    Args:
        df: DataFrame com dados de múltiplos anos
        reference_year: Ano de referência
        current_year: Ano atual para comparação
        feature_cols: Features para verificar
        ano_col: Nome da coluna de ano

    Returns:
        Resultado do check_drift
    """
    ref = df[df[ano_col] == reference_year]
    cur = df[df[ano_col] == current_year]

    if ref.empty or cur.empty:
        logger.warning(f"Dados insuficientes: ref={len(ref)}, cur={len(cur)}")
        return {"dataset_drift": False, "error": "Dados insuficientes"}

    logger.info(f"Comparando drift: {reference_year} ({len(ref)} registros) → {current_year} ({len(cur)} registros)")

    return check_drift(ref, cur, feature_cols)
