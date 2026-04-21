"""
bossfx.utils.logger
===================

Production-grade logging. Uses stdlib only (no external deps) but is
designed so swapping in ``structlog`` or shipping to Datadog later
requires no code changes in callers.
"""
from __future__ import annotations

import logging
import os
import sys
from logging import Logger

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Call once at application startup."""
    global _configured
    if _configured:
        return

    level_int = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))

    root = logging.getLogger("bossfx")
    root.setLevel(level_int)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> Logger:
    """Get a namespaced logger. Convention: use ``__name__`` at module top."""
    if not _configured:
        configure_logging(os.environ.get("BOSSFX_LOG_LEVEL", "INFO"))
    return logging.getLogger(name)
