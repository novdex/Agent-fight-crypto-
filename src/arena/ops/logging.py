"""Structured logging for the arena: one "arena" root logger, optional
JSON-lines output (one JSON object per record) for log shippers.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_LOGGER_NAME = "arena"

# Stashed on the LogRecord under private keys so structured fields can never
# collide with reserved LogRecord attributes (msg, args, name, ...).
_EVENT_ATTR = "_arena_event"
_FIELDS_ATTR = "_arena_fields"


class _JsonLinesFormatter(logging.Formatter):
    """Render each record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {}
        fields = getattr(record, _FIELDS_ATTR, None)
        if isinstance(fields, dict):
            payload.update(fields)
        # Core keys win over user fields so the line shape stays predictable.
        payload["ts"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload["level"] = record.levelname
        payload["logger"] = record.name
        payload["event"] = getattr(record, _EVENT_ATTR, None) or record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            payload["error"] = repr(record.exc_info[1])
        return json.dumps(payload, default=str)


class _PlainFormatter(logging.Formatter):
    """Human-readable formatter that appends ``key=value`` structured fields."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = getattr(record, _FIELDS_ATTR, None)
        if isinstance(fields, dict) and fields:
            base += " " + " ".join(f"{k}={v}" for k, v in fields.items())
        return base


def setup_logging(json_mode: bool = False) -> None:
    """Configure the root ``"arena"`` logger (idempotent).

    With ``json_mode`` every record is emitted as one JSON line; otherwise a
    conventional human-readable line is used. Output goes to stderr.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonLinesFormatter() if json_mode else _PlainFormatter())
    logger.addHandler(handler)


def log_event(event: str, **fields: object) -> None:
    """Emit one structured INFO record on the ``"arena"`` logger."""
    logging.getLogger(_LOGGER_NAME).info(
        event, extra={_EVENT_ATTR: event, _FIELDS_ATTR: fields}
    )
