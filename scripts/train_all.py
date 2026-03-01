#!/usr/bin/env python3
"""
Script principal de treinamento de todos os modelos.

Este script orquestra:
1. Carregamento e harmonização dos dados
2. Feature engineering
3. Treinamento do classificador de risco
4. Treinamento do modelo de clusterização
5. Salvamento dos modelos
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_all_years, compare_schemas, get_common_columns
from src.data.preprocessing import harmonize_datasets, filter_common_features_only, normalize_pedra_column
from src.data.feature_engineering import create_all_temporal_features, create_target_variable
from src.models.train_classifier import train_risk_classifier
from src.models.train_clustering import train_student_clustering


def setup_logging():
    """Configura logging."""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")


def main():
    """Função principal."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Treina todos os modelos do Passos Mágicos")
    parser.add_argument("--no-mlflow", action="store_true", help="Desabilita MLflow tracking")
    parser.add_argument("--skip-classifier", action="store_true", help="Pula treinamento do classificador")
    parser.add_argument("--skip-clustering", action="store_true", help="Pula treinamento da clusterização")
    args = parser.parse_args()

    use_mlflow = not args.no_mlflow

    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO DE TODOS OS MODELOS")
    logger.info("=" * 60)

    # 1. Carregar dados
    logger.info("\n[1/6] Carregando dados...")
    datasets = load_all_years()

    if len(datasets) < 2:
        logger.error("É necessário pelo menos 2 anos de dados para treinar o classificador")
        logger.info("Para treinar apenas clusterização, use --skip-classifier")
        if len(datasets) == 0:
            logger.error("Nenhum dataset encontrado!")
            return 1

    logger.info(f"Datasets carregados: {list(datasets.keys())}")

    # 2. Comparar schemas (informativo)
    logger.info("\n[2/6] Analisando schemas...")
    schema_comp = compare_schemas(datasets)
    common_cols = get_common_columns(datasets)
    logger.info(f"Colunas comuns: {len(common_cols)}")
    logger.info(f"Colunas comuns: {common_cols}")

    # 3. Harmonizar datasets
    logger.info("\n[3/6] Harmonizando datasets...")
    df = harmonize_datasets(
        df_2022=datasets.get(2022),
        df_2023=datasets.get(2023),
        df_2024=datasets.get(2024),
    )
    logger.info(f"Total de registros após harmonização: {len(df)}")

    # 4. Feature Engineering
    logger.info("\n[4/6] Aplicando feature engineering...")

    # Normaliza coluna de pedra
    df = normalize_pedra_column(df)

    # Cria features temporais
    df = create_all_temporal_features(df)

    # Cria target para classificação
    df = create_target_variable(df)

    logger.info(f"Features criadas. Total de colunas: {len(df.columns)}")

    # 5. Treinar Classificador de Risco
    if not args.skip_classifier:
        logger.info("\n[5a/6] Treinando classificador de risco...")

        # Features para classificação
        classifier_features = [
            "INDE", "IEG", "IDA", "IPS", "IAA",
            "delta_INDE", "delta_IEG", "delta_IDA",
            "anos_no_programa", "tendencia_INDE",
            "pedras_mudadas_total"
        ]
        classifier_features = [f for f in classifier_features if f in df.columns]

        try:
            classifier_model, classifier_metrics = train_risk_classifier(
                df,
                classifier_features,
                use_mlflow=use_mlflow
            )
            logger.info(f"Classificador treinado! F1: {classifier_metrics['f1']:.3f}")
        except Exception as e:
            logger.error(f"Erro ao treinar classificador: {e}")
            if "not enough" in str(e) or "empty" in str(e):
                logger.warning("Não há dados suficientes para treinar o classificador")
    else:
        logger.info("\n[5a/6] Classificador pulado (--skip-classifier)")

    # 6. Treinar Clusterização
    if not args.skip_clustering:
        logger.info("\n[5b/6] Treinando modelo de clusterização...")

        # Usa dados mais recentes de cada aluno
        df_latest = df.sort_values("ano").groupby("RA").last().reset_index()

        # Features para clusterização
        clustering_features = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        clustering_features = [f for f in clustering_features if f in df_latest.columns]

        # Remove NaN
        df_cluster = df_latest.dropna(subset=clustering_features)
        logger.info(f"Dados para clusterização: {len(df_cluster)} alunos")

        try:
            clustering_model, scaler, clustering_metrics = train_student_clustering(
                df_cluster,
                clustering_features,
                n_clusters=4,
                use_mlflow=use_mlflow
            )
            logger.info(f"Clusterização treinada! Silhouette: {clustering_metrics['silhouette_score']:.3f}")
        except Exception as e:
            logger.error(f"Erro ao treinar clusterização: {e}")
    else:
        logger.info("\n[5b/6] Clusterização pulada (--skip-clustering)")

    # 7. Resumo
    logger.info("\n[6/6] Treinamento concluído!")
    logger.info("=" * 60)
    logger.info("Modelos salvos em:")
    logger.info("  - models/classifier.pkl")
    logger.info("  - models/clustering_model.pkl")
    logger.info("  - models/scaler.pkl")
    logger.info("  - models/cluster_labels.json")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
