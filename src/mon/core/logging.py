#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python's :mod`logging` module."""

from __future__ import annotations

__all__ = [
    "disable_print",
    "enable_print",
    "get_logger",
    "logger",
]

import logging
import os
import sys

from rich import logging as r_logging

from mon.core import pathlib

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
        path: The path to store the log info. Default: ``None``.
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


# region Print

def disable_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__

# endregion
