#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extend Rich API: https://github.com/willmcgugan/rich
"""

from __future__ import annotations

import inspect
import sys

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
from rich.progress import TransferSpeedColumn
from rich.table import Column
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from one.core.device import get_gpu_memory
from one.core.types import assert_dict
from one.core.types import assert_list_of
from one.core.types import MemoryUnit
from one.core.types import MemoryUnit_


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
    """
    It returns a progress bar that displays the current time, the task
    description, a progress bar, the percentage complete, the transfer speed,
    the amount downloaded, the time remaining, the time elapsed, and a
    right-pointing arrow.
    
    Returns:
        A progress bar.
    """
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
    """
    It returns a progress bar that displays the current time, the task
    description, a progress bar, the percentage complete, the number of items
    processed, the processing speed, the time remaining, the time elapsed,
    and a spinner.
    
    Returns:
        A progress bar.
    """
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
    """
    It takes a dictionary and prints it in a pretty format.
    
    Args:
        data (dict): The data to be printed.
        title (str): The title of the panel.
    """
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
    """
    This function takes a list of dictionaries and prints them in a table.
    
    Args:
        data (list[dict]): A list of dictionary.
    """
    assert_list_of(data, dict)
    table = Table(show_header=True, header_style="bold magenta")
    for k, v in data[0].items():
        table.add_column(k)
    
    for d in data:
        row = [f"{v}" for v in d.values()]
        table.add_row(*row)
    
    console.log(table)


@dispatch(dict)
def print_table(data: dict):
    """
    It takes a dictionary and prints it as a table.
    
    Args:
        data (dict): dict
    """
    assert_dict(data)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Key")
    table.add_column("Value")

    for k, v in data.items():
        row = [f"{k}", f"{v}"]
        table.add_row(*row)

    console.log(table)


# MARK: - Modules

class GPUMemoryUsageColumn(ProgressColumn):
    """
    Renders GPU memory usage, e.g. `33.1/48.0GB`.
    """

    def __init__(
        self,
        unit        : MemoryUnit_   = MemoryUnit.GB,
        table_column: Column | None = None
    ):
        """
        This function initializes a `Memory` object.
        
        Args:
            unit (MemoryUnit_): The unit of memory to use.
            table_column (Column | None): The column in the table that this
                field is associated with.
        """
        super().__init__(table_column=table_column)
        self.unit = MemoryUnit.from_value(unit)

    def render(self, task: Task) -> Text:
        """
        It returns a Text object with the memory usage of the GPU.
        
        Args:
            task (Task): Task.
        
        Returns:
            A Text object with the memory status.
        """
        total, used, free = get_gpu_memory()
        memory_status     = f"{used:.1f}/{total:.1f}{self.unit.value}"
        memory_text       = Text(memory_status, style="bright_yellow")
        return memory_text
    

class ProcessedItemsColumn(ProgressColumn):
    """
    Renders processed files and total, e.g. `1728/2025`.
    """

    def __init__(self, table_column: Column | None = None):
        """
        This function is a constructor for the class `Column`.
        
        Args:
            table_column (Column | None): The column that this widget is
                associated with.
        """
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        """
        It takes a Task object and returns a Text object.
        
        Args:
            task (Task): Task.
        
        Returns:
            A Text object with the download status.
        """
        completed       = int(task.completed)
        total           = int(task.total)
        download_status = f"{completed}/{total}"
        download_text   = Text(download_status, style="progress.download")
        return download_text


class ProcessingSpeedColumn(ProgressColumn):
    """
    Renders human-readable progress speed.
    """

    def render(self, task: Task) -> Text:
        """
        It takes a task and returns a Text object.
        
        Args:
            task (Task): Task.
        
        Returns:
            A Text object with the speed data.
        """
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        speed_data = "{:.2f}".format(speed)
        return Text(f"{speed_data}it/s", style="progress.data.speed")


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
