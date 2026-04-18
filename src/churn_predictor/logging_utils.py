from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging format for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_event(
    logger: logging.Logger,
    message: str,
    *,
    event: str,
    model: str | None = None,
    step: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit log with consistent structured fields."""
    payload = {
        "message": message,
        "event": event,
        "model": model or "n/a",
        "step": step or "n/a",
    }
    logger.log(level, str(payload))
