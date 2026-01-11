#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Logging utility and context manager collection.

This module provides logger creation and configuration, noisy library logger
management, and a context manager for suppressing console output.
"""

from __future__ import annotations

__all__ = [
    "OutputSuppressor",
    "disable_print",
    "enable_print",
    "get_logger",
    "logger",
]

import logging
import os
import sys

from rich import logging as r_logging

from mon.core.pathlib import Path


# Set a default log level for TensorFlow to reduce verbosity on import.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# Configure the root logger to use RichHandler for pretty, colorful logging.
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [r_logging.RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def get_logger(
    path: Path | str = None,
    name: str        = "global_logger"
) -> logging.Logger:
    """Return a configured logger.

    Create a logger and, if ``path`` is provided, attach a file handler that
    writes INFO-level records with timestamps and file and line context.

    Args:
        path: Optional path to a log file. If ``path`` is None, file logging is
            skipped. Defaults to None.
        name: Name of the logger. Defaults to "global_logger".

    Returns:
        Configured logger instance.
    """
    lgr = logging.getLogger(name)

    if path:
        path = str(path)
        # Prevent duplicate handlers for the same file.
        if not any(
            isinstance(h, logging.FileHandler) and
            h.baseFilename == os.path.abspath(path)
            for h in lgr.handlers
        ):
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            formatter    = logging.Formatter(
                "%(asctime)s [%(filename)s:%(lineno)s] %(levelname)s: %(message)s"
            )
            file_handler.setFormatter(formatter)
            lgr.addHandler(file_handler)

    return lgr

# endregion


# ==============================================================================
# region CONTROL
# ==============================================================================

class OutputSuppressor:
    """Stdout and stderr redirection state manager.

    Manage the state of stdout and stderr redirection.

    Attributes:
        _original_stdout (TextIO): Original stdout stream.
        _original_stderr (TextIO): Original stderr stream.
        _devnull (TextIO | None): File handle for /dev/null. Defaults to None.
    """

    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    _devnull         = None

    @classmethod
    def disable(cls):
        """Redirect stdout and stderr to /dev/null."""
        if cls._devnull is None:
            cls._devnull = open(os.devnull, "w")
        sys.stdout = cls._devnull
        sys.stderr = cls._devnull

    @classmethod
    def enable(cls):
        """Restore original stdout and stderr."""
        sys.stdout = cls._original_stdout
        sys.stderr = cls._original_stderr
        # Note: We keep _devnull open to avoid re-opening overhead.


def _enable_default_loggers():
    """Restore default logger levels to INFO.

    Reset the logging levels for common libraries to restore standard logging
    behavior.
    """
    noisy_loggers = [None, "torch", "tensorflow"]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.INFO)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def _disable_default_loggers():
    """Silence noisy library loggers.

    Set the level of noisy library loggers to a high value to reduce console
    noise.
    """
    # Using a high integer value to ensure only critical errors are logged.
    SILENCE_LEVEL = 50

    # A list of common libraries that produce verbose output.
    # `None` refers to the root logger.
    noisy_loggers = [
        None, "torch", "tensorflow", "tensorboard", "mmcv", "fsspec", "urllib3"
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(SILENCE_LEVEL)
    # Set TF environment variable as a backup for subprocesses.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Silence noisy libraries on import to provide a cleaner default experience.
_disable_default_loggers()


def enable_print():
    """Restore all console output and logging levels."""
    OutputSuppressor.enable()
    _enable_default_loggers()


def disable_print():
    """Completely silence the console."""
    OutputSuppressor.disable()
    _disable_default_loggers()

# endregion
