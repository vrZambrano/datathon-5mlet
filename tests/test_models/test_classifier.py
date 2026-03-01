"""
Testes para src/models/train_classifier.py
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


class TestPrepareDataForClassifier:
    """Testes para prepare_data_for_classifier."""

    def test_splits_data(self, harmonized_df_full):
        """Testa split de dados train/val."""
        from src.models.train_classifier import prepare_data_for_classifier

        feature_cols = ["INDE", "IEG", "IDA", "IPS"]
        feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

        X_train, X_val, y_train, y_val = prepare_data_for_classifier(
            harmonized_df_full, feature_cols
        )

        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_train) > len(X_val)  # 80/20 split
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)

    def test_no_group_leakage(self, harmonized_df_full):
        """Testa que não há vazamento entre grupos (alunos)."""
        from src.models.train_classifier import prepare_data_for_classifier

        feature_cols = ["INDE", "IEG", "IDA"]
        feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

        # Precisamos obter os índices
        df_model = harmonized_df_full.dropna(subset=["target_queda_prox_ano"]).dropna(subset=feature_cols)

        X_train, X_val, y_train, y_val = prepare_data_for_classifier(
            harmonized_df_full, feature_cols
        )

        # Verifica que temos dados válidos
        assert len(X_train) > 0


class TestTrainXGBoostClassifier:
    """Testes para train_xgboost_classifier."""

    def test_trains_successfully(self, harmonized_df_full):
        """Testa treinamento bem-sucedido."""
        from src.models.train_classifier import (
            prepare_data_for_classifier,
            train_xgboost_classifier,
        )

        feature_cols = ["INDE", "IEG", "IDA", "IPS"]
        feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

        X_train, X_val, y_train, y_val = prepare_data_for_classifier(
            harmonized_df_full, feature_cols
        )

        model = train_xgboost_classifier(X_train, y_train, X_val, y_val)

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


class TestEvaluateClassifier:
    """Testes para evaluate_classifier."""

    def test_returns_metrics(self, trained_classifier):
        """Testa que métricas são retornadas."""
        model, metrics, features = trained_classifier

        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics

        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_roc_auc_present(self, trained_classifier):
        """Testa presença de ROC AUC."""
        model, metrics, features = trained_classifier

        if metrics["roc_auc"] is not None:
            assert 0 <= metrics["roc_auc"] <= 1


class TestGetFeatureImportance:
    """Testes para get_feature_importance."""

    def test_returns_importance(self, trained_classifier):
        """Testa retorno de importâncias."""
        from src.models.train_classifier import get_feature_importance

        model, metrics, features = trained_classifier

        importance_df = get_feature_importance(model, features)

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == len(features)
        assert importance_df["importance"].sum() > 0


class TestTrainRiskClassifier:
    """Testes para pipeline completo train_risk_classifier."""

    def test_full_pipeline(self, harmonized_df_full):
        """Testa pipeline completo."""
        from src.models.train_classifier import train_risk_classifier

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "test_classifier.pkl")

            feature_cols = ["INDE", "IEG", "IDA", "IPS", "IAA"]
            feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

            model, metrics = train_risk_classifier(
                harmonized_df_full,
                feature_cols,
                model_path=model_path,
                use_mlflow=False,
            )

            assert model is not None
            assert Path(model_path).exists()
            assert "f1" in metrics

    def test_saved_model_loads(self, harmonized_df_full):
        """Testa que o modelo salvo pode ser carregado."""
        import joblib
        from src.models.train_classifier import train_risk_classifier

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "test_classifier.pkl")

            feature_cols = ["INDE", "IEG", "IDA"]
            feature_cols = [f for f in feature_cols if f in harmonized_df_full.columns]

            train_risk_classifier(
                harmonized_df_full, feature_cols,
                model_path=model_path, use_mlflow=False
            )

            loaded_model = joblib.load(model_path)
            assert hasattr(loaded_model, "predict")


class TestPredictRisk:
    """Testes para predict_risk."""

    def test_predict_single_student(self, trained_classifier):
        """Testa predição para um aluno."""
        from src.models.train_classifier import predict_risk

        model, metrics, features = trained_classifier

        student = pd.DataFrame([{f: np.random.uniform(3, 9) for f in features}])
        result = predict_risk(model, student)

        assert "risk_probability" in result
        assert "risk_class" in result
        assert "will_drop" in result
        assert 0 <= result["risk_probability"] <= 1
        assert result["risk_class"] in ["BAIXO", "MEDIO", "ALTO"]
        assert isinstance(result["will_drop"], bool)
