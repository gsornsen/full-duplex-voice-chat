"""Structured logging utilities."""

import json
import logging
from typing import Any


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Setup structured logging.

    Args:
        level: Logging level
        json_format: Whether to use JSON format
    """
    logging.basicConfig(level=getattr(logging, level.upper()))


def log_event(event_type: str, data: dict[str, Any]) -> None:
    """Log structured event.

    Args:
        event_type: Event type identifier
        data: Event data dictionary
    """
    logging.info(json.dumps({"event": event_type, **data}))
