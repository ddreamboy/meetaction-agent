import sys
from pathlib import Path

from loguru import logger


def setup_logging() -> None:
    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
        level="INFO",
        diagnose=False,
        backtrace=False,
    )

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "meetaction.jsonl",
        serialize=True,
        rotation="10 MB",
        retention=5,
        enqueue=True,
        diagnose=False,
        backtrace=False,
    )

    logger.info("Logger initialized")
