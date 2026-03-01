"""
Testes para src/models/train_clustering.py
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path


class TestPrepareDataForClustering:
    """Testes para prepare_data_for_clustering."""

    def test_scales_data(self, harmonized_df):
        """Testa escalamento dos dados."""
        from src.models.train_clustering import prepare_data_for_clustering

        feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
        feature_cols = [f for f in feature_cols if f in harmonized_df.columns]

        df_clean = harmonized_df.dropna(subset=feature_cols)
        X_scaled, scaler = prepare_data_for_clustering(df_clean, feature_cols)

        assert X_scaled.shape[1] == len(feature_cols)
        assert scaler is not None

        # Dados escalados devem ter média ~0
        for col in X_scaled.columns:
            assert abs(X_scaled[col].mean()) < 0.5

    def test_with_existing_scaler(self, harmonized_df):
        """Testa com scaler já fornecido."""
        from sklearn.preprocessing import StandardScaler
        from src.models.train_clustering import prepare_data_for_clustering

        feature_cols = ["INDE", "IEG"]
        feature_cols = [f for f in feature_cols if f in harmonized_df.columns]

        df_clean = harmonized_df.dropna(subset=feature_cols)

        # Primeira chamada cria scaler
        X1, scaler = prepare_data_for_clustering(df_clean, feature_cols)

        # Segunda chamada usa scaler existente
        X2, scaler2 = prepare_data_for_clustering(df_clean, feature_cols, scaler=scaler)

        assert (X1.values == X2.values).all()


class TestFindOptimalClusters:
    """Testes para find_optimal_clusters."""

    def test_returns_scores(self, harmonized_df):
        """Testa que retorna scores para cada k."""
        from src.models.train_clustering import prepare_data_for_clustering, find_optimal_clusters

        feature_cols = ["INDE", "IEG", "IDA"]
        feature_cols = [f for f in feature_cols if f in harmonized_df.columns]

        df_clean = harmonized_df.dropna(subset=feature_cols)
        X_scaled, _ = prepare_data_for_clustering(df_clean, feature_cols)

        scores = find_optimal_clusters(X_scaled, max_clusters=5)

        assert isinstance(scores, dict)
        assert len(scores) > 0
        # Keys should be cluster numbers
        assert all(isinstance(k, int) for k in scores.keys())
        # Values should be silhouette scores
        assert all(-1 <= v <= 1 for v in scores.values())


class TestTrainKMeans:
    """Testes para train_kmeans."""

    def test_trains_kmeans(self, harmonized_df):
        """Testa treinamento de K-Means."""
        from src.models.train_clustering import prepare_data_for_clustering, train_kmeans

        feature_cols = ["INDE", "IEG", "IDA"]
        feature_cols = [f for f in feature_cols if f in harmonized_df.columns]

        df_clean = harmonized_df.dropna(subset=feature_cols)
        X_scaled, _ = prepare_data_for_clustering(df_clean, feature_cols)

        model = train_kmeans(X_scaled, n_clusters=4)

        assert model is not None
        assert hasattr(model, "predict")
        assert len(np.unique(model.labels_)) == 4


class TestEvaluateClustering:
    """Testes para evaluate_clustering."""

    def test_returns_metrics(self, trained_clustering):
        """Testa retorno de métricas."""
        model, scaler, metrics, features = trained_clustering

        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "n_clusters" in metrics

        assert -1 <= metrics["silhouette_score"] <= 1
        assert metrics["davies_bouldin_score"] > 0
        assert metrics["n_clusters"] == 4


class TestNameClusters:
    """Testes para name_clusters."""

    def test_names_four_clusters(self):
        """Testa nomeação de 4 clusters."""
        from src.models.train_clustering import name_clusters

        clusters_info = {
            0: {"INDE_mean": 3.0, "IEG_mean": 2.5},
            1: {"INDE_mean": 5.0, "IEG_mean": 4.5},
            2: {"INDE_mean": 7.0, "IEG_mean": 6.5},
            3: {"INDE_mean": 9.0, "IEG_mean": 8.5},
        }

        names = name_clusters(clusters_info)

        assert len(names) == 4
        assert all(isinstance(v, str) for v in names.values())
        # Menor INDE deve ser "Desmotivados Crônicos"
        assert "Desmotivados" in names[0]
        # Maior INDE deve ser "Alto Desempenho"
        assert "Alto" in names[3]


class TestTrainStudentClustering:
    """Testes para pipeline completo train_student_clustering."""

    def test_full_pipeline(self, harmonized_df):
        """Testa pipeline completo."""
        from src.models.train_clustering import train_student_clustering

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "kmeans.pkl")
            scaler_path = str(Path(tmpdir) / "scaler.pkl")
            labels_path = str(Path(tmpdir) / "labels.json")

            feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
            feature_cols = [f for f in feature_cols if f in harmonized_df.columns]

            df_clean = harmonized_df.dropna(subset=feature_cols)

            model, scaler, metrics = train_student_clustering(
                df_clean,
                feature_cols,
                n_clusters=4,
                model_path=model_path,
                scaler_path=scaler_path,
                labels_path=labels_path,
                use_mlflow=False,
            )

            assert Path(model_path).exists()
            assert Path(scaler_path).exists()
            assert Path(labels_path).exists()

            # Verifica labels JSON
            with open(labels_path) as f:
                data = json.load(f)
            assert "cluster_names" in data
            assert "n_clusters" in data

    def test_saved_model_loads(self, harmonized_df):
        """Testa que modelo salvo pode ser carregado."""
        import joblib
        from src.models.train_clustering import train_student_clustering

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "kmeans.pkl")
            scaler_path = str(Path(tmpdir) / "scaler.pkl")
            labels_path = str(Path(tmpdir) / "labels.json")

            feature_cols = ["INDE", "IEG", "IDA"]
            feature_cols = [f for f in feature_cols if f in harmonized_df.columns]

            df_clean = harmonized_df.dropna(subset=feature_cols)

            train_student_clustering(
                df_clean, feature_cols, n_clusters=3,
                model_path=model_path, scaler_path=scaler_path,
                labels_path=labels_path, use_mlflow=False
            )

            loaded = joblib.load(model_path)
            assert hasattr(loaded, "predict")
            assert len(np.unique(loaded.labels_)) == 3


class TestPredictCluster:
    """Testes para predict_cluster."""

    def test_predict_single_student(self, trained_clustering):
        """Testa predição para um aluno."""
        from src.models.train_clustering import predict_cluster

        model, scaler, metrics, features = trained_clustering

        student = pd.DataFrame([{f: np.random.uniform(3, 9) for f in features}])
        result = predict_cluster(model, scaler, student)

        assert "cluster_id" in result
        assert "cluster_name" in result
        assert isinstance(result["cluster_id"], int)
        assert 0 <= result["cluster_id"] < 4
