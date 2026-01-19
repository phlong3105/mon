#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Enhanced console logging and pretty-printing utilities.

This module provides rich Console instances and helpers for logging and rendering
structured data.
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

import box
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

rich_console_theme = Theme({
    "debug"    : "dark_green",
    "info"     : "green",
    "warning"  : "yellow",
    "error"    : "bright_red",
    "critical" : "bold red",
})

console = Console(
    color_system    = "auto",
    log_time_format = "[%X]",
    soft_wrap       = True,
    width           = None,
    theme           = rich_console_theme,
)

error_console = Console(
    color_system    = "auto",
    log_time_format = "[%X]",
    soft_wrap       = False,
    width           = None,
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)

# --- Shortcuts ---

log       = console.log
log_error = error_console.log

# endregion


# ==============================================================================
# region DEBUGGING
# ==============================================================================

# --- Basic Logging ---

def pprint_dict(a_dict: dict | box.Box, title: str = ""):
    """Pretty-print a mapping inside a panel.

    Args:
        a_dict: Mapping to print.
        title: Optional title for the panel. Defaults to "".

    Raises:
        TypeError: If ``a_dict`` is not a dict or box.Box.
    """
    a_dict = to_dict(a_dict)
    # Create a Pretty object for structured rendering
    pr     = pretty.Pretty(
        a_dict,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold",
    )
    p      = Panel(pr, title=title)
    console.log(p)


def rprint_dict(a_dict: dict | box.Box, title: str = ""):
    """Render a mapping as a two-column table.

    Args:
        a_dict: Mapping to print.
        title: Optional table title. Defaults to "".

    Raises:
        TypeError: If ``a_dict`` is not a dict or box.Box.
    """
    a_dict = to_dict(a_dict)
    # Initialize a two-column table
    tab    = Table(
        title        = title,
        show_header  = True,
        row_styles   = ["dim", ""],
        header_style = "bold magenta",
        highlight    = True,
    )
    tab.add_column("Key"   , justify="left")
    tab.add_column("Value" , justify="left")

    # Let rich handle rendering of keys and values for better formatting.
    for k, v in a_dict.items():
        tab.add_row(str(k), v)
    console.log(tab)


def rprint_list_dicts(list_of_dicts: list[dict]):
    """Render a list of dictionaries as a table with shared columns.

    Args:
        list_of_dicts: List of dictionaries that must share identical keys.

    Raises:
        ValueError: If ``list_of_dicts`` is not a non-empty list, or if the
            dictionaries do not share identical keys.
    """
    if not isinstance(list_of_dicts, list) or not list_of_dicts:
        raise ValueError(
            f"Expected 'list_of_dicts' to be a non-empty list, "
            f"but got {type(list_of_dicts).__name__}."
        )

    # Extract headers from the first dictionary and create a set for quick key
    # comparison.
    headers    = list(list_of_dicts[0].keys())
    header_set = set(headers)
    tab        = Table(
        show_header  = True,
        header_style = "bold magenta",
    )

    for k in headers:
        tab.add_column(str(k), no_wrap=True)

    for d in list_of_dicts:
        if set(d.keys()) != header_set:
            raise ValueError(
                f"All dicts must have the same keys. Expected keys {header_set}, "
                f"but got {set(d.keys())} in dict: {d}."
            )
        # Let rich handle rendering of values for better formatting.
        tab.add_row(*(d[k] for k in headers))

    console.log(tab)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
