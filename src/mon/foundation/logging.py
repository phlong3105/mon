#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python's :mod`logging` module."""

from __future__ import annotations

__all__ = [
    "get_logger", "logger",
]

import logging

from rich import logging as r_logging

from mon.foundation import pathlib

# region Logging

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [r_logging.RichHandler(rich_tracebacks = True)]
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)


def get_logger(path: pathlib.Path | None = None) -> logging.Logger:
    """Get access the global :param:`logging.Logger` object that uses
    :mod:`rich`. Create a new one if it doesn't exist.
    
    Args:
        path: The path to store the log info. Defaults to None.
    """
    if path:
        path = logging.FileHandler(path)
        path.setLevel(logging.INFO)
        path.setFormatter(logging.Formatter(
            " %(asctime)s [%(file_name)s %(lineno)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(path)
    
    return logger

# endregion
