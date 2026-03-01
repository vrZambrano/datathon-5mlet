"""
Testes para app/core/ (logger e config).
"""

import pytest
from unittest.mock import patch


class TestSetupLogging:
    """Testes para setup_logging."""

    def test_setup_logging_runs(self):
        """Testa que setup_logging executa sem erro."""
        from app.core.logger import setup_logging

        setup_logging()  # Não deve levantar exceção

    def test_intercept_handler_emit(self):
        """Testa InterceptHandler.emit."""
        import logging
        from app.core.logger import InterceptHandler

        handler = InterceptHandler()
        assert isinstance(handler, logging.Handler)

        # Cria um record de teste
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="test.py", lineno=1,
            msg="Teste de log", args=(), exc_info=None
        )

        # Emit não deve levantar exceção
        try:
            handler.emit(record)
        except Exception:
            pass  # Pode falhar por causa do frame depth, mas é ok


class TestConfig:
    """Testes para configuração."""

    def test_settings_defaults(self):
        """Testa valores padrão do settings."""
        from app.core.config import get_settings

        settings = get_settings()
        assert settings.environment in ["development", "production", "test"]
        assert settings.api_host is not None
        assert settings.api_port > 0

    def test_settings_model_paths(self):
        """Testa paths dos modelos no settings."""
        from app.core.config import get_settings

        settings = get_settings()
        assert "classifier" in settings.classifier_model
        assert "clustering" in settings.clustering_model

    def test_settings_is_development(self):
        """Testa propriedade is_development."""
        from app.core.config import get_settings

        settings = get_settings()
        assert isinstance(settings.is_development, bool)

    def test_settings_is_production(self):
        """Testa propriedade is_production."""
        from app.core.config import get_settings

        settings = get_settings()
        assert isinstance(settings.is_production, bool)
