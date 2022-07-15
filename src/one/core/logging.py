#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import Union

from rich.logging import RichHandler

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

def get_logger(log_file: Union[str, None] = None):
    """Create logger object.
    
    Args:
        log_file (str, None):
            Provide the log file to save logging info. Default: `None`.
    """
    # NOTE: File Logging
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
