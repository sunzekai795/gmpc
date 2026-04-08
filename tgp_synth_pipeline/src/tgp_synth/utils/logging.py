from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging.

    If `rich` is available, use a RichHandler; otherwise, fallback to stdlib.
    """

    root = logging.getLogger()
    if root.handlers:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)

    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)
        logging.basicConfig(
            level=numeric,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[handler],
        )
    except Exception:
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
