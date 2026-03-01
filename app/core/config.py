"""
Configurações da aplicação FastAPI.
Utiliza pydantic-settings para carregar variáveis de ambiente.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Configurações da aplicação."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/passos_magicos"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str = "minio"
    aws_secret_access_key: str = "minio123"

    # OpenRouter (LLM)
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_fallback_model: str = "meta-llama/llama-3.1-70b"

    # Model Configuration
    model_path: str = "models"
    classifier_model: str = "models/classifier.pkl"
    clustering_model: str = "models/clustering_model.pkl"
    scaler: str = "models/scaler.pkl"

    # Feature Engineering
    risk_threshold: float = 0.7
    n_clusters: int = 4

    # Monitoring
    drift_threshold: float = 0.5
    enable_mlflow: bool = True

    @property
    def is_development(self) -> bool:
        """Verifica se está em modo desenvolvimento."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Verifica se está em modo produção."""
        return self.environment == "production"


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna instância cacheada das configurações.

    Returns:
        Settings: Configurações da aplicação
    """
    return Settings()
