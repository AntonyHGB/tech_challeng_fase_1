from __future__ import annotations

import json
import logging


class JSONFormatter(logging.Formatter):
    """Formatador de logs para saída JSON estruturada."""

    def format(self, record: logging.LogRecord) -> str:
        """Formata o registro de log iterando sobre os atributos padrão e extras."""
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Ignora campos padrões internos do LogRecord
        builtin_keys = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

        # Adiciona extras
        for key, value in record.__dict__.items():
            if key not in builtin_keys:
                payload[key] = value

        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging format for the project."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(level)


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
    logger.log(
        level,
        message,
        extra={
            "event": event,
            "model": model or "n/a",
            "step": step or "n/a",
        },
    )
