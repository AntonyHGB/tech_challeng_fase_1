from __future__ import annotations

import logging
from typing import Optional


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
    model: Optional[str] = None,
    step: Optional[str] = None,
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
