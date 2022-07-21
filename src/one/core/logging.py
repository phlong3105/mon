#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import logging
import sys

from rich.logging import RichHandler

from one.core.types import Path

field_style = {
    "asctime"  : {"color": "green"},
    "levelname": {"bold" : True},
    "filename" : {"color": "cyan"},
    "funcName" : {"color": "blue"}
}

level_styles = {
    "critical": {"bold" : True, "color": "red"},
    "debug"   : {"color": "green"},
    "error"   : {"color": "red"},
    "info"    : {"color": "magenta"},
    "warning" : {"color": "yellow"}
}


logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)


# MARK: - Functional

def get_logger(log_file: Path | None = None):
    """
    It creates a logger that logs to a file if a file is provided.
    
    Args:
        log_file (Path | None): The given log file.
    
    Returns:
        A logger object.
    """
    if log_file:
        file = logging.FileHandler(log_file)
        file.setLevel(logging.INFO)
        file.setFormatter(logging.Formatter(
            " %(asctime)s [%(filename)s %(lineno)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(file)

    return logger


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
