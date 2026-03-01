"""
Fixtures compartilhadas para todos os testes.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Adiciona dir raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES DE DADOS
# ============================================================================

@pytest.fixture
def sample_df_2022():
    """DataFrame simulando dados de 2022."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "RA": [f"RA-{i}" for i in range(1, n + 1)],
        "NOME": [f"Aluno-{i}" for i in range(1, n + 1)],
        "INDE 22": np.random.uniform(3, 10, n),
        "IEG": np.random.uniform(2, 10, n),
        "IDA": np.random.uniform(2, 10, n),
        "IPS": np.random.uniform(2, 10, n),
        "IAA": np.random.uniform(2, 10, n),
        "IAN": np.random.uniform(2, 10, n),
        "IPP": np.random.uniform(2, 10, n),
        "IPV": np.random.uniform(2, 10, n),
        "PEDRA 22": np.random.choice(
            ["Quartzo", "Ágata", "Ametista", "Topázio"], n
        ),
        "FASE": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n),
        "TURMA": np.random.choice(["A", "B", "C"], n),
    })


@pytest.fixture
def sample_df_2023():
    """DataFrame simulando dados de 2023."""
    np.random.seed(43)
    n = 60
    return pd.DataFrame({
        "RA": [f"RA-{i}" for i in range(1, n + 1)],
        "INDE 2023": np.random.uniform(3, 10, n).astype(str),
        "IEG": np.random.uniform(2, 10, n),
        "IDA": np.random.uniform(2, 10, n),
        "IPS": np.random.uniform(2, 10, n),
        "IAA": np.random.uniform(2, 10, n),
        "IAN": np.random.uniform(2, 10, n),
        "IPP": np.random.uniform(2, 10, n),
        "IPV": np.random.uniform(2, 10, n),
        "PEDRA 2023": np.random.choice(
            ["Quartzo", "Ágata", "Ametista", "Topázio"], n
        ),
    })


@pytest.fixture
def sample_df_2024():
    """DataFrame simulando dados de 2024."""
    np.random.seed(44)
    n = 70
    return pd.DataFrame({
        "RA": [f"RA-{i}" for i in range(1, n + 1)],
        "INDE 2024": np.random.uniform(3, 10, n).astype(str),
        "IEG": np.random.uniform(2, 10, n),
        "IDA": np.random.uniform(2, 10, n),
        "IPS": np.random.uniform(2, 10, n),
        "IAA": np.random.uniform(2, 10, n),
        "IAN": np.random.uniform(2, 10, n),
        "IPP": np.random.uniform(2, 10, n),
        "IPV": np.random.uniform(2, 10, n),
        "PEDRA 2024": np.random.choice(
            ["Quartzo", "Ágata", "Ametista", "Topázio"], n
        ),
    })


@pytest.fixture
def harmonized_df(sample_df_2022, sample_df_2023, sample_df_2024):
    """DataFrame harmonizado com todos os anos."""
    from src.data.preprocessing import harmonize_datasets, normalize_pedra_column

    df = harmonize_datasets(
        df_2022=sample_df_2022,
        df_2023=sample_df_2023,
        df_2024=sample_df_2024,
    )
    df = normalize_pedra_column(df)
    return df


@pytest.fixture
def harmonized_df_with_features(harmonized_df):
    """DataFrame harmonizado com features temporais."""
    from src.data.feature_engineering import create_all_temporal_features

    return create_all_temporal_features(harmonized_df)


@pytest.fixture
def harmonized_df_full(harmonized_df_with_features):
    """DataFrame completo com target."""
    from src.data.feature_engineering import create_target_variable

    return create_target_variable(harmonized_df_with_features)


# ============================================================================
# FIXTURES DE MODELOS
# ============================================================================

@pytest.fixture
def trained_classifier(harmonized_df_full):
    """Treina e retorna um classificador de risco."""
    from src.models.train_classifier import train_risk_classifier

    features = ["INDE", "IEG", "IDA", "IPS", "IAA",
                "delta_INDE", "delta_IEG", "delta_IDA",
                "anos_no_programa", "tendencia_INDE", "pedras_mudadas_total"]
    features = [f for f in features if f in harmonized_df_full.columns]

    model, metrics = train_risk_classifier(
        harmonized_df_full, features, use_mlflow=False
    )
    return model, metrics, features


@pytest.fixture
def trained_clustering(harmonized_df):
    """Treina e retorna um modelo de clusterização."""
    from src.models.train_clustering import train_student_clustering

    features = ["INDE", "IEG", "IDA", "IPS", "IAA"]
    features = [f for f in features if f in harmonized_df.columns]

    df_clean = harmonized_df.dropna(subset=features)

    model, scaler, metrics = train_student_clustering(
        df_clean, features, n_clusters=4, use_mlflow=False
    )
    return model, scaler, metrics, features


# ============================================================================
# FIXTURES DE API
# ============================================================================

@pytest.fixture
def api_client():
    """Cliente de teste para FastAPI."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def mock_settings():
    """Settings mockadas para testes."""
    from app.core.config import Settings

    return Settings(
        environment="test",
        openrouter_api_key="test-key-12345678901234567890",
        classifier_model="models/classifier.pkl",
        clustering_model="models/clustering_model.pkl",
        scaler="models/scaler.pkl",
    )
