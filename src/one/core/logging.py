#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler

__all__ = [
    "logger",
    "get_logger",
]

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

def get_logger(log_file: Optional[str] = None):
    """Create logger object.
    
    Args:
        log_file (str, optional):
            Provide the log file to save logging info.
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
