"""
Treinamento do modelo de clusterização de perfis de alunos.

Este módulo treina um K-Means para agrupar alunos em perfis similares
para ações personalizadas.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger

# MLflow (opcional)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow não disponível")


def prepare_data_for_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Prepara dados para clusterização (escalamento).

    Args:
        df: DataFrame com features
        feature_cols: Lista de colunas para clusterização
        scaler: Scaler já ajustado (opcional)

    Returns:
        Tuple (DataFrame escalado, Scaler)
    """
    # Seleciona features
    X = df[feature_cols].copy()

    # Remove linhas com NaN
    X = X.dropna()

    logger.info(f"Dados para clusterização: {X.shape}")

    # Escala features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled, scaler


def find_optimal_clusters(
    X: pd.DataFrame,
    max_clusters: int = 10
) -> Dict[int, float]:
    """
    Encontra número ótimo de clusters usando Elbow Method e Silhouette.

    Args:
        X: DataFrame escalado com features
        max_clusters: Máximo número de clusters a testar

    Returns:
        Dicionário com número de clusters -> silhouette score
    """
    logger.info("Buscando número ótimo de clusters...")

    scores = {}
    inertias = []

    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        sil_score = silhouette_score(X, labels)
        scores[n] = sil_score
        inertias.append(kmeans.inertia_)

        logger.info(f"  n_clusters={n}, silhouette={sil_score:.3f}")

    # Melhor silhouette
    best_n = max(scores, key=scores.get)
    logger.info(f"Melhor número de clusters: {best_n} (silhouette={scores[best_n]:.3f})")

    return scores


def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> KMeans:
    """
    Treina modelo K-Means.

    Args:
        X: DataFrame escalado com features
        n_clusters: Número de clusters
        random_state: Random state

    Returns:
        Modelo K-Means treinado
    """
    logger.info(f"Treinando K-Means com {n_clusters} clusters...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        max_iter=300
    )

    kmeans.fit(X)

    logger.info("K-Means treinado com sucesso")
    return kmeans


def evaluate_clustering(
    X: pd.DataFrame,
    labels: np.ndarray
) -> Dict:
    """
    Avalia qualidade da clusterização.

    Args:
        X: Dados escalados
        labels: Labels dos clusters

    Returns:
        Dicionário com métricas
    """
    # Silhouette Score
    sil_score = silhouette_score(X, labels)

    # Davies-Bouldin Index (menor é melhor)
    db_score = davies_bouldin_score(X, labels)

    # Inércia
    inertia = None

    metrics = {
        "silhouette_score": float(sil_score),
        "davies_bouldin_score": float(db_score),
        "n_clusters": int(len(np.unique(labels))),
    }

    logger.info(f"Silhouette: {sil_score:.3f}, Davies-Bouldin: {db_score:.3f}")

    return metrics


def analyze_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    scaler: StandardScaler
) -> Dict[int, Dict]:
    """
    Analisa as características de cada cluster.

    Args:
        df: DataFrame original (não escalado)
        labels: Labels dos clusters
        feature_cols: Colunas usadas na clusterização
        scaler: Scaler usado (para desscalcular se necessário)

    Returns:
        Dicionário com informações de cada cluster
    """
    df_analysis = df.copy()
    df_analysis["cluster"] = labels

    clusters_info = {}

    for cluster_id in sorted(df_analysis["cluster"].unique()):
        cluster_df = df_analysis[df_analysis["cluster"] == cluster_id]

        info = {
            "n_students": len(cluster_df),
            "percentage": len(cluster_df) / len(df_analysis) * 100,
        }

        # Estatísticas das features
        for col in feature_cols:
            if col in cluster_df.columns:
                info[f"{col}_mean"] = float(cluster_df[col].mean())
                info[f"{col}_std"] = float(cluster_df[col].std())

        clusters_info[cluster_id] = info

    return clusters_info


def name_clusters(clusters_info: Dict[int, Dict]) -> Dict[int, str]:
    """
    Atribui nomes descritivos aos clusters baseado nas características.

    Args:
        clusters_info: Informações dos clusters

    Returns:
        Dicionário cluster_id -> nome
    """
    # Análise simples baseada em INDE médio e IEG médio
    cluster_names = {}

    inde_means = {cid: info.get("INDE_mean", 0) for cid, info in clusters_info.items()}
    ieg_means = {cid: info.get("IEG_mean", 0) for cid, info in clusters_info.items()}

    # Ordena clusters por INDE
    sorted_by_inde = sorted(inde_means.items(), key=lambda x: x[1])

    # Atribui nomes baseado em posição
    if len(sorted_by_inde) >= 4:
        # Menor INDE
        cluster_names[sorted_by_inde[0][0]] = "Desmotivados Crônicos"
        cluster_names[sorted_by_inde[1][0]] = "Em Risco"
        cluster_names[sorted_by_inde[2][0]] = "Engajados com Dificuldade"
        # Maior INDE
        cluster_names[sorted_by_inde[3][0]] = "Alto Desempenho"
    else:
        for i, (cid, _) in enumerate(sorted_by_inde):
            cluster_names[cid] = f"Cluster {i + 1}"

    return cluster_names


def train_student_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 4,
    model_path: str = "models/clustering_model.pkl",
    scaler_path: str = "models/scaler.pkl",
    labels_path: str = "models/cluster_labels.json",
    use_mlflow: bool = True
) -> Tuple[KMeans, StandardScaler, Dict]:
    """
    Pipeline completo de treinamento da clusterização.

    Args:
        df: DataFrame com features
        feature_cols: Lista de colunas para clusterização
        n_clusters: Número de clusters
        model_path: Caminho para salvar o modelo
        scaler_path: Caminho para salvar o scaler
        labels_path: Caminho para salvar labels dos clusters
        use_mlflow: Se deve usar MLflow para tracking

    Returns:
        Tuple (modelo_kmeans, scaler, métricas)
    """
    logger.info("=" * 50)
    logger.info("Iniciando treinamento da clusterização")
    logger.info("=" * 50)

    # Prepara dados
    X_scaled, scaler = prepare_data_for_clustering(df, feature_cols)

    # Treina K-Means
    kmeans = train_kmeans(X_scaled, n_clusters)

    # Avalia
    labels = kmeans.labels_
    metrics = evaluate_clustering(X_scaled, labels)

    # Analisa clusters
    # Usa df original com índice correspondente
    df_for_analysis = df.loc[X_scaled.index].copy()
    clusters_info = analyze_clusters(df_for_analysis, labels, feature_cols, scaler)

    # Nomeia clusters
    cluster_names = name_clusters(clusters_info)
    logger.info(f"Nomes dos clusters: {cluster_names}")

    # Salva modelo
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, model_path)
    logger.info(f"Modelo salvo em: {model_path}")

    # Salva scaler
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler salvo em: {scaler_path}")

    # Salva labels
    labels_data = {
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
        "cluster_info": {str(k): v for k, v in clusters_info.items()},
        "n_clusters": n_clusters,
        "feature_cols": feature_cols
    }
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)
    logger.info(f"Labels salvos em: {labels_path}")

    # MLflow tracking
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="student_clustering"):
                mlflow.log_params({
                    "n_clusters": n_clusters,
                    "feature_cols": ",".join(feature_cols)
                })

                mlflow.log_metrics({
                    "silhouette_score": metrics["silhouette_score"],
                    "davies_bouldin_score": metrics["davies_bouldin_score"]
                })

                mlflow.sklearn.log_model(kmeans, "model")

                logger.info("Experimento logado no MLflow")
        except Exception as e:
            logger.warning(f"Erro ao logar no MLflow: {e}")

    return kmeans, scaler, metrics


def predict_cluster(
    model: KMeans,
    scaler: StandardScaler,
    student_features: pd.DataFrame
) -> Dict:
    """
    Faz predição do cluster para um aluno.

    Args:
        model: Modelo K-Means treinado
        scaler: Scaler ajustado
        student_features: DataFrame com features do aluno

    Returns:
        Dicionário com cluster e informações
    """
    # Escala features
    X_scaled = scaler.transform(student_features)

    # Prediz cluster
    cluster_id = int(model.predict(X_scaled)[0])

    # Carrega nomes dos clusters
    try:
        with open("models/cluster_labels.json", "r") as f:
            labels_data = json.load(f)
        cluster_name = labels_data["cluster_names"].get(str(cluster_id), f"Cluster {cluster_id}")
    except:
        cluster_name = f"Cluster {cluster_id}"

    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
    }


if __name__ == "__main__":
    # Teste do módulo
    from src.data.loader import load_all_years
    from src.data.preprocessing import harmonize_datasets, filter_common_features_only, normalize_pedra_column
    from src.data.feature_engineering import create_all_temporal_features

    # Carrega dados
    datasets = load_all_years()

    if len(datasets) >= 1:
        # Harmoniza
        df = harmonize_datasets(**datasets)

        # Normaliza pedra
        df = normalize_pedra_column(df)

        # Usa dados mais recentes de cada aluno
        df_latest = df.sort_values("ANO").groupby("RA").last().reset_index()

        # Features para clusterização (apenas do ano atual)
        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        feature_cols = [f for f in feature_cols if f in df_latest.columns]

        # Remove NaN
        df_cluster = df_latest.dropna(subset=feature_cols)

        # Treina
        kmeans, scaler, metrics = train_student_clustering(
            df_cluster, feature_cols, n_clusters=4, use_mlflow=False
        )

        print("\nModelo treinado com sucesso!")
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
    else:
        print("Carregue pelo menos 1 ano de dados para treinar o modelo")
