"""
Treinamento do classificador de risco de queda de pedra.

Este módulo treina um modelo XGBoost para prever se um aluno vai cair
de nível (pedra) no ano seguinte.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger

# MLflow (opcional, pode não estar instalado)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow não disponível")


def prepare_data_for_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "target_queda_prox_ano",
    id_col: str = "RA"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepara dados para treino do classificador.

    Usa GroupShuffleSplit para não vazamento de dados (mesmo aluno não em treino e teste).

    Args:
        df: DataFrame com features e target
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna target
        id_col: Coluna de ID do aluno

    Returns:
        Tuple (X_train, X_val, y_train, y_val)
    """
    # Remove linhas com target NaN
    df_model = df.dropna(subset=[target_col]).copy()

    # Remove linhas com features críticas NaN
    df_model = df_model.dropna(subset=feature_cols)

    logger.info(f"Dados para classificação: {len(df_model)} registros")

    # Seleciona features
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()
    groups = df_model[id_col].copy()

    # Split por grupos (alunos) para evitar vazamento
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    logger.info(f"Split: Treino={len(X_train)}, Val={len(X_val)}")
    logger.info(f"Target - Treino: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")

    return X_train, X_val, y_train, y_val


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None
) -> xgb.XGBClassifier:
    """
    Treina classificador XGBoost.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        y_val: Target de validação
        params: Parâmetros do XGBoost (opcional)

    Returns:
        Modelo XGBoost treinado
    """
    if params is None:
        params = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": 3,  # Para lidar com desbalanceamento
        }

    model = xgb.XGBClassifier(**params)

    logger.info("Treinando classificador XGBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def evaluate_classifier(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict:
    """
    Avalia o classificador e retorna métricas.

    Args:
        model: Modelo treinado
        X_val: Features de validação
        y_val: Target de validação

    Returns:
        Dicionário com métricas
    """
    # Predições
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Métricas
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_val, y_pred_proba)
    except:
        roc_auc = None

    metrics = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "target_distribution": {
            "positive": int(y_val.sum()),
            "negative": int(len(y_val) - y_val.sum())
        }
    }

    logger.info(f"Métricas: F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f}")
    if roc_auc:
        logger.info(f"ROC AUC: {roc_auc:.3f}")

    return metrics


def get_feature_importance(model: xgb.XGBClassifier, feature_names: list) -> pd.DataFrame:
    """
    Retorna importância das features.

    Args:
        model: Modelo XGBoost treinado
        feature_names: Lista de nomes das features

    Returns:
        DataFrame com importâncias
    """
    importance = model.feature_importances_

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return df_importance


def train_risk_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "target_queda_prox_ano",
    id_col: str = "RA",
    model_path: str = "models/classifier.pkl",
    use_mlflow: bool = True
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Pipeline completo de treinamento do classificador de risco.

    Args:
        df: DataFrame com features e target
        feature_cols: Lista de colunas de features
        target_col: Nome da coluna target
        id_col: Coluna de ID do aluno
        model_path: Caminho para salvar o modelo
        use_mlflow: Se deve usar MLflow para tracking

    Returns:
        Tuple (modelo_treinado, métricas)
    """
    logger.info("=" * 50)
    logger.info("Iniciando treinamento do classificador de risco")
    logger.info("=" * 50)

    # Prepara dados
    X_train, X_val, y_train, y_val = prepare_data_for_classifier(
        df, feature_cols, target_col, id_col
    )

    # Treina modelo
    model = train_xgboost_classifier(X_train, y_train, X_val, y_val)

    # Avalia
    metrics = evaluate_classifier(model, X_val, y_val)

    # Importância de features
    feature_importance = get_feature_importance(model, feature_cols)
    logger.info("\nTop 10 Features:\n" + feature_importance.head(10).to_string())

    # Salva modelo
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Modelo salvo em: {model_path}")

    # MLflow tracking
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name="risk_classifier"):
                # Log parâmetros
                mlflow.log_params(model.get_params())

                # Log métricas
                mlflow.log_metrics({
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                })
                if metrics["roc_auc"]:
                    mlflow.log_metric("roc_auc", metrics["roc_auc"])

                # Log modelo
                mlflow.sklearn.log_model(model, "model")

                logger.info("Experimento logado no MLflow")
        except Exception as e:
            logger.warning(f"Erro ao logar no MLflow: {e}")

    return model, metrics


def predict_risk(
    model: xgb.XGBClassifier,
    student_features: pd.DataFrame
) -> Dict:
    """
    Faz predição de risco para um aluno.

    Args:
        model: Modelo treinado
        student_features: DataFrame com features do aluno

    Returns:
        Dicionário com risco e informações
    """
    # Predição
    risk_proba = model.predict_proba(student_features)[0, 1]
    risk_class = model.predict(student_features)[0]

    # Classificação do risco
    if risk_proba >= 0.7:
        risk_level = "ALTO"
    elif risk_proba >= 0.4:
        risk_level = "MEDIO"
    else:
        risk_level = "BAIXO"

    return {
        "risk_probability": float(risk_proba),
        "risk_class": risk_level,
        "will_drop": bool(risk_class),
    }


if __name__ == "__main__":
    # Teste do módulo
    from src.data.loader import load_all_years
    from src.data.preprocessing import harmonize_datasets, filter_common_features_only, normalize_pedra_column
    from src.data.feature_engineering import create_all_temporal_features, create_target_variable

    # Carrega dados
    datasets = load_all_years()

    if len(datasets) >= 2:
        # Harmoniza
        df = harmonize_datasets(**datasets)

        # Normaliza pedra
        df = normalize_pedra_column(df)

        # Features temporais
        df = create_all_temporal_features(df)

        # Cria target
        df = create_target_variable(df)

        # Features para modelo
        feature_cols = [
            "INDE", "IEG", "IDA", "IPS",
            "delta_INDE", "delta_IEG", "delta_IDA",
            "anos_no_programa", "tendencia_INDE"
        ]
        feature_cols = [f for f in feature_cols if f in df.columns]

        # Treina
        model, metrics = train_risk_classifier(
            df, feature_cols, use_mlflow=False
        )

        print("\nModelo treinado com sucesso!")
        print(f"F1 Score: {metrics['f1']:.3f}")
    else:
        print("Carregue pelo menos 2 anos de dados para treinar o modelo")
