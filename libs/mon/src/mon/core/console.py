#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Console & Logging.

This module extends the ``rich.console`` module with custom consoles and
logging utilities.
"""

from __future__ import annotations

__all__ = [
    "clear_terminal",
    "console",
    "disable_print",
    "enable_print",
    "error_console",
    "log",
    "log_error",
    "pprint_dict",
    "rprint_dict",
    "rprint_list_dicts",
]

import logging
import os
import platform
import sys

from box import Box
from rich import pretty
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from mon.core.utils import is_list_of

# ==============================================================================
# region CONSTANTS
# ==============================================================================

_rich_console_theme = Theme(
    {
        "debug": "dark_green",
        "info": "green",
        "warning": "yellow",
        "error": "bright_red",
        "critical": "bold red",
    },
)

console = Console(
    color_system="auto",
    log_time_format="[%X]",
    soft_wrap=True,
    width=None,
    theme=_rich_console_theme,
)

error_console = Console(
    color_system="auto",
    log_time_format="[%X]",
    soft_wrap=False,
    width=None,
    stderr=True,
    style="bold red",
    theme=_rich_console_theme,
)

log = console.log
log_error = error_console.log

# endregion


# ==============================================================================
# region CONTROL
# ==============================================================================

# --- Terminal Control ---

def clear_terminal():
    """Clear the terminal screen."""
    if platform.system() == "Windows":
        # For Windows, 'cls' is the standard command.
        os.system("cls")
    else:
        # For POSIX systems, use ANSI escape codes for efficiency.
        # \033[H moves the cursor to the top-left corner.
        # \033[2J clears the entire screen.
        print("\033[H\033[2J", end="", flush=True)


# --- Logging Control ---

class OutputSuppressor:
    """Manage the state of stdout and stderr redirection.

    Attributes:
        original_stdout(sys.stdout): Original stdout stream.
        original_stderr(sys.stderr): Original stderr stream.
        devnull(file): File descriptor for /dev/null.
    """

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    devnull = None

    @classmethod
    def disable(cls):
        """Redirect stdout and stderr to /dev/null."""
        if cls.devnull is None:
            cls.devnull = open(os.devnull, "w")
        sys.stdout = cls.devnull
        sys.stderr = cls.devnull

    @classmethod
    def enable(cls):
        """Restore original stdout and stderr."""
        sys.stdout = cls.original_stdout
        sys.stderr = cls.original_stderr
        # Note: We keep _devnull open to avoid re-opening overhead.


def _enable_default_loggers():
    """Restore default logger levels to INFO.

    Reset the logging levels for common libraries to restore standard logging
    behavior.
    """
    # A list of common libraries that produce verbose output.
    # `None` refers to the root logger.
    noisy_loggers = [
        None, "torch", "tensorflow", "tensorboard", "mmcv", "fsspec", "urllib3",
    ]
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
        None, "torch", "tensorflow", "tensorboard", "mmcv", "fsspec", "urllib3",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(SILENCE_LEVEL)

    # Set TF environment variable as a backup for subprocesses.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def enable_print():
    """Restore all console output and logging levels."""
    OutputSuppressor.enable()
    _enable_default_loggers()


def disable_print():
    """Completely silence the console."""
    OutputSuppressor.disable()
    _disable_default_loggers()


# Silence noisy libraries on import to provide a cleaner default experience.
_disable_default_loggers()

# endregion


# ==============================================================================
# region BASIC LOGGING
# ==============================================================================

def pprint_dict(value: dict, title: str = ""):
    """Pretty-print a dictionary inside a ``rich`` Panel.

    Args:
        value (dict): Dictionary to print.
        title (str, optional): Panel title. Defaults to "".
    """
    # Normalize inputs
    value = value.to_dict() if isinstance(value, Box) else value

    # Create a Pretty object for structured rendering
    pr = pretty.Pretty(
        value,
        expand_all=True,
        indent_guides=True,
        insert_line=True,
        overflow="fold",
    )
    p = Panel(pr, title=title)
    console.log(p)


def rprint_dict(value: dict, title: str = ""):
    """Pretty-print a dictionary as a two-column table.

    Args:
        value (dict): Dictionary to print.
        title (str, optional): Panel title. Defaults to "".
    """
    # Normalize inputs
    value = value.to_dict() if isinstance(value, Box) else value

    # Initialize a two-column table
    tab = Table(
        title=title,
        show_header=True,
        row_styles=["dim", ""],
        header_style="bold magenta",
        highlight=True,
    )
    tab.add_column("Key", justify="left")
    tab.add_column("Value", justify="left")

    # Let rich handle rendering of keys and values for better formatting.
    for k, v in value.items():
        tab.add_row(str(k), v)
    console.log(tab)


def rprint_list_dicts(values: list[dict]):
    """Pretty-print a list of dictionaries as a table with shared columns.

    Args:
        values (list[dict]): List of dictionaries to print.s

    Raises:
        ValueError: If ``values`` is not a non-empty list, or if the dictionaries
            do not share identical keys.
    """
    if not is_list_of(values, dict):
        raise ValueError(
            f"Expected 'values' to be a non-empty list, "
            f"but got: '{type(values).__name__}'.",
        )

    # Extract headers from the first dictionary and create a set for quick key
    # comparison
    table = Table(show_header=True, header_style="bold magenta")
    headers = list(values[0].keys())
    header_set = set(headers)

    for k in headers:
        table.add_column(str(k), no_wrap=True)

    for d in values:
        if set(d.keys()) != header_set:
            raise ValueError(
                f"All dicts must have the same keys. Expected keys "
                f"{header_set}, but got: {set(d.keys())} in dict: {d}.",
            )
        # Let rich handle rendering of values for better formatting.
        table.add_row(*(d[k] for k in headers))

    console.log(table)

# endregion
