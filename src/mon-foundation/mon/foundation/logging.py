#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python's :mod`logging` module."""

from __future__ import annotations

__all__ = [
    "get_logger", "logger",
]

import logging

from rich import logging as r_logging

from mon.foundation.typing import PathType

# region Logging

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [r_logging.RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)


def get_logger(log_file: PathType | None = None) -> logging.Logger:
    """Get access the global :param:`logging.Logger` object that uses
    :mod:`rich`. Create a new one if it doesn't exist.
    
    Args:
        log_file: The path to store the log info. Defaults to None.
    """
    if log_file:
        file = logging.FileHandler(log_file)
        file.setLevel(logging.INFO)
        file.setFormatter(logging.Formatter(
            " %(asctime)s [%(filename)s %(lineno)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(file)

    return logger

# endregion
