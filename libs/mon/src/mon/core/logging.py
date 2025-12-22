#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Logging utility and context manager collection.

This module provides logger creation and configuration, noisy library logger
management, and context managers for suppressing stdout to enable consistent
logging and output control across the codebase.
"""

__all__ = [
    "disable_print",
    "enable_print",
    "logger",
]

import contextlib
import logging
import os
import sys
from typing import Iterator

from rich import logging as r_logging

from mon.core.pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ==============================================================================
# CORE LOGGING SYSTEM
# ==============================================================================

# --- Initialization (Configuring the global Rich handler) ---
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [r_logging.RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
# logger.setLevel(logging.INFO)


# --- Factories ---
def get_logger(path: Path = None) -> logging.Logger:
    """Return a configured logger.

    Create a logger and, if a path is provided, attach a file handler that
    writes INFO-level records with timestamps and file and line context.

    Args:
        path: Optional path to a logfile. If None, skip file logging.
    """
    logger = logging.getLogger("global_logger")
    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


# ==============================================================================
# OUTPUT INTERCEPTION
# ==============================================================================

# --- Context Managers ---
@contextlib.contextmanager
def _disable_stdout() -> Iterator[None]:
    """Suppress stdout temporarily.

    Redirect stdout to os.devnull for the duration of the context manager.

    Yields:
        Use the context to suppress print output for the enclosed block.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _enable_stdout():
    """Restore stdout to the original stream.

    Reset sys.stdout back to the original standard output stream.
    """
    sys.stdout = sys.__stdout__


# --- Global Toggle ---
def disable_print():
    """Disable printing and silence default loggers.

    Disable printing to stdout and silence common noisy library loggers to
    reduce console clutter.
    """
    _disable_stdout()
    _disable_default_loggers()


def enable_print():
    """Enable printing and restore logger levels.

    Restore stdout and re-enable the default logger levels that were previously
    suppressed.
    """
    _enable_stdout()
    _enable_default_loggers()


# ==============================================================================
# OUTPUT INTERCEPTION
# ==============================================================================

# --- Third-Party Silencers (Specific handlers for TF/Torch/Built-ins) ---
def _disable_default_loggers():
    """Silence noisy library loggers.

    Set global and frequently noisy library loggers to a high level to
    suppress log record emission and reduce console noise.
    """
    # Disabling Python’s Built-in logging
    logging.getLogger().setLevel(logging.CRITICAL + 1)  # Suppresses everything
    # Disabling PyTorch Logs
    logging.getLogger("torch").setLevel(logging.CRITICAL + 1)  # Silence PyTorch logs
    # Disabling TensorFlow Logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _enable_default_loggers():
    """Restore default logger levels.

    Reset global and common library logger levels and the TensorFlow
    environment variable to restore standard logging behavior.
    """
    # Enabling Python’s Built-in logging
    logging.getLogger().setLevel(logging.INFO)  # Restores default level
    # Enabling PyTorch Logs
    logging.getLogger("torch").setLevel(logging.INFO)  # Restores default level
    # Enabling TensorFlow Logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Restores default level

# Disable default loggers
_disable_default_loggers()
