#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for enhanced console logging and pretty-printing.

This module provides utilities for improved console logging and pretty-printing
using the ``rich`` library. It includes functions to log messages with different
severity levels, and to pretty-print dictionaries and lists of dictionaries in
a visually appealing format.
"""

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


# ----- Console -----
rich_console_theme = Theme({
    "debug"   : "dark_green",
    "info"    : "green",
    "warning" : "yellow",
    "error"   : "bright_red",
    "critical": "bold red",
})

console = Console(
    color_system    = "auto",
    log_time_format = "[%X]",  # "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = True,
    width           = None,  # 120,
    theme           = rich_console_theme,
)

error_console = Console(
    color_system    = "auto",
    log_time_format = "[%X]",  # "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = False,
    width           = None,  # 120,
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)


# ----- Pretty Print -----
log       = console.log
log_error = error_console.log


def pprint_dict(a_dict: dict | box.Box, title: str = ""):
    """Prints a dictionary with a title using pretty.Pretty format.

    Args:
        a_dict (dict or box.Box): Dictionary to print.
        title (str): Title above the printed dictionary. Defaults to "".

    Raises:
        TypeError: If ``a_dict`` is not a dictionary.
    """
    if isinstance(a_dict, box.Box):
        a_dict = a_dict.to_dict()
    if not isinstance(a_dict, dict):
        raise TypeError(f"``a_dict`` must be a dict, got {type(a_dict)}.")
    pr = pretty.Pretty(
        a_dict,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    p = Panel(pr, title=f"{title}")
    console.log(p)


def rprint_dict(a_dict: dict | box.Box, title: str = ""):
    """Prints a dictionary in a table format.

    Args:
        a_dict (dict or box.Box): Dictionary to print.
        title (str): Title above the printed dictionary. Defaults to "".

    Raises:
        TypeError: If ``a_dict`` is not a dictionary.
    """
    if isinstance(a_dict, box.Box):
        a_dict = a_dict.to_dict()
    if not isinstance(a_dict, dict):
        raise TypeError(f"``a_dict`` must be a dict, got {type(a_dict)}.")
    tab = Table(
        title        = title,
        show_header  = True,
        row_styles   = ["dim", ""],
        header_style = "bold magenta",
        highlight    = True,
    )
    tab.add_column("Key")
    tab.add_column("Value")
    for k, v in a_dict.items():
        row = [f"{k}", f"{v}"]
        tab.add_row(*row)
    console.log(tab)


def rprint_list_dicts(list_of_dicts: list[dict]):
    """Prints a list of similar dictionaries in a table format.
    
    Args:
        list_of_dicts (list of dict): List of dictionaries to print.
            All dictionaries must have identical keys.

    Raises:
        TypeError: If ``list_of_dicts`` is not a list of dictionaries.
        ValueError: If ``list_of_dicts`` is empty or if the dictionaries
            do not have identical keys.
    """
    if not isinstance(list_of_dicts, list) or not all(isinstance(d, dict) for d in list_of_dicts):
        raise TypeError(f"``list_of_dicts`` must be a list of dicts, got {type(list_of_dicts)}.")
    if not list_of_dicts:
        raise ValueError("``list_of_dicts`` must not be empty.")
    if not all(set(d.keys()) == set(list_of_dicts[0].keys()) for d in list_of_dicts):
        raise ValueError("All dictionaries in ``list_of_dicts`` must have identical keys.")
    tab = Table(show_header=True, header_style="bold magenta")
    for k in list_of_dicts[0].keys():
        tab.add_column(k, no_wrap=True)
    for d in list_of_dicts:
        row = [f"{v}" for v in d.values()]
        tab.add_row(*row)
    console.log(tab)
