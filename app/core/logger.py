"""
Configuração de logging para a aplicação.
"""

import sys
from loguru import logger
from app.core.config import get_settings


def setup_logging() -> None:
    """
    Configura o logging da aplicação usando loguru.
    """
    settings = get_settings()

    # Remove handler padrão
    logger.remove()

    # Adiciona handler para stdout
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # Adiciona handler para arquivo (apenas em produção ou se configurado)
    if settings.is_production:
        logger.add(
            "logs/app.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    logger.info(f"Logging configurado - Nível: {settings.log_level}, Ambiente: {settings.environment}")


# Para compatibilidade com logging padrão Python
class InterceptHandler:
    """
    Intercepta logs do logging padrão e redireciona para loguru.
    """

    def emit(self, record):
        """Emite log usando loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )
