#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extend Rich API: https://github.com/willmcgugan/rich
"""

from typing import Optional

from multipledispatch import dispatch
from munch import Munch
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import SpinnerColumn
from rich.progress import Task
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import track
from rich.progress import TransferSpeedColumn
from rich.table import Column
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from one.core.collection import is_list_of
from one.core.device import get_gpu_memory
from one.core.device import MemoryUnit

__all__ = [
    "console",
    "error_console",
    "download_bar",
    "print_dict",
    "print_table",
    "progress_bar",
    "track",
    "Console",
    "GPUMemoryUsageColumn",
    "ProcessedItemsColumn",
    "ProcessingSpeedColumn",
    "Table",
]


# MARK: - Globals

custom_theme = Theme({
    "debug"   : "dark_green",
    "info"    : "green",
    "warning" : "yellow",
    "error"   : "bright_red",
    "critical": "bold red",
})

console = Console(
    color_system    = "windows",
    log_time_format = "[%x %H:%M:%S:%f]",
    soft_wrap       = True,
    theme           = custom_theme,
)

error_console = Console(
    color_system    = "windows",
    log_time_format = "[%x %H:%M:%S:%f]",
    soft_wrap       = True,
    stderr          = True,
    style           = "bold red",
    theme           = custom_theme,
)


# MARK: - Functional

def download_bar() -> Progress:
    return Progress(
        TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
            justify="left", style="log.time"
        ),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
        ">",
        TimeElapsedColumn(),
        console=console,
    )


def progress_bar() -> Progress:
    return Progress(
        TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
            justify="left", style="log.time"
        ),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None, finished_style="green"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        ProcessedItemsColumn(),
        "•",
        ProcessingSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        ">",
        TimeElapsedColumn(),
        SpinnerColumn(),
        console=console,
    )


def print_dict(data: dict, title: str = ""):
    """Print a dictionary."""
    if isinstance(data, Munch):
        data = data.toDict()
        
    pretty = Pretty(
        data,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    panel = Panel(pretty, title=f"{title}")
    console.log(panel)
    

@dispatch(list)
def print_table(data: list[dict]):
    """Print a list of dictionary as a table."""
    if not is_list_of(data, dict):
        raise ValueError(f"`data` must be a `list[dict]`. But got: {type(data)}.")
    
    table = Table(show_header=True, header_style="bold magenta")
    for k, v in data[0].items():
        table.add_column(k)
    
    for d in data:
        row = [f"{v}" for v in d.values()]
        table.add_row(*row)
    
    console.log(table)


@dispatch(dict)
def print_table(data: dict):
    """Print a dictionary as a table."""
    if not isinstance(data, dict):
        raise ValueError(f"`data` must be a `dict`. But got: {type(data)}.")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Key")
    table.add_column("Value")

    for k, v in data.items():
        row = [f"{k}", f"{v}"]
        table.add_row(*row)

    console.log(table)


# MARK: - Modules

class GPUMemoryUsageColumn(ProgressColumn):
    """Renders GPU memory usage, e.g. `33.1/48.0GB`."""

    def __init__(
        self,
        unit        : MemoryUnit       = MemoryUnit.GB,
        table_column: Optional[Column] = None
    ):
        super().__init__(table_column=table_column)
        self.unit = unit

    def render(self, task: Task) -> Text:
        """Calculate common unit for completed and total."""
        total, used, free = get_gpu_memory()
        memory_status     = f"{used:.1f}/{total:.1f}{self.unit.value}"
        memory_text       = Text(memory_status, style="bright_yellow")
        return memory_text
    

class ProcessedItemsColumn(ProgressColumn):
    """Renders processed files and total, e.g. `1728/2025`."""

    def __init__(self, table_column: Optional[Column] = None):
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        """Calculate common unit for completed and total."""
        completed       = int(task.completed)
        total           = int(task.total)
        download_status = f"{completed}/{total}"
        download_text   = Text(download_status, style="progress.download")
        return download_text


class ProcessingSpeedColumn(ProgressColumn):
    """Renders human-readable progress speed."""

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        speed_data = "{:.2f}".format(speed)
        return Text(f"{speed_data}it/s", style="progress.data.speed")
