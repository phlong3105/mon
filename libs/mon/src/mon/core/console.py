#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Enhanced console logging and pretty-printing utilities.

This module provides rich Console instances and helpers for logging and
rendering structured data.
"""

from __future__ import annotations

__all__ = [
    "console",
    "error_console",
    "log",
    "log_error",
    "pprint_dict",
    "rprint_dict",
    "rprint_list_dicts",
]

from box import Box
from rich import pretty
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from mon.core.utils import to_dict

# ==============================================================================
# region CONSTANTS
# ==============================================================================

# --- Defaults ---

rich_console_theme = Theme(
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
    theme=rich_console_theme,
)

error_console = Console(
    color_system="auto",
    log_time_format="[%X]",
    soft_wrap=False,
    width=None,
    stderr=True,
    style="bold red",
    theme=rich_console_theme,
)

# --- Shortcuts ---

log = console.log
log_error = error_console.log


# endregion


# ==============================================================================
# region DEBUGGING
# ==============================================================================

# --- Basic Logging ---

def pprint_dict(value: dict, title: str = ""):
    """Pretty-print a dictionary inside a ``rich`` Panel.

    Args:
        value (Box | dict): Dictionary to print.
        title (str): Panel title. Defaults to "".
    """
    # value = to_dict(value)
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
        title (str): Panel title. Defaults to "".
    """
    # value = to_dict(value)
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
        values (list[dict]): List of dictionaries to print.

    Raises:
        ValueError: If ``values`` is not a non-empty list, or if the dictionaries
            do not share identical keys.
    """
    if not isinstance(values, list) or any(not isinstance(d, dict) for d in values):
        raise ValueError(
            f"Expected 'values' to be a non-empty list, "
            f"but got {type(values).__name__}.",
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
                f"{header_set}, but got {set(d.keys())} in dict: {d}.",
            )
        # Let rich handle rendering of values for better formatting.
        table.add_row(*(d[k] for k in headers))

    console.log(table)


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
