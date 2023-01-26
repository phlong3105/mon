#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the :mod:`rich` package.

It provides rich text and beautiful formatting in the terminal, console, and
logging throughout the :mod:`mon` framework.
"""

from __future__ import annotations

import multipledispatch
import munch
import rich
from rich import panel, pretty, progress, table, text, theme

from mon.core import constant, pynvml
from mon.core.typing import MemoryUnitType

# region Console

field_style = {
    "asctime"  : {"color": "green"},
    "levelname": {"bold" : True   },
    "file_name" : {"color": "cyan" },
    "funcName" : {"color": "blue" }
}

level_styles = {
    "critical": {"bold" : True, "color": "red"},
    "debug"   : {"color": "green"             },
    "error"   : {"color": "red"               },
    "info"    : {"color": "magenta"           },
    "warning" : {"color": "yellow"            }
}

rich_console_theme = theme.Theme(
    {
        "debug"   : "dark_green",
        "info"    : "green",
        "warning" : "yellow",
        "error"   : "bright_red",
        "critical": "bold red",
    }
)

console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%m/%d/%Y %H:%M:%S.%f]",
    soft_wrap       = False,
    width           = 200,
    theme           = rich_console_theme,
)

error_console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%m/%d/%Y %H:%M:%S.%f]",
    soft_wrap       = False,
    width           = 200,
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)


def get_console() -> rich.console.Console:
    """Get access to the global :class:`rich.console.Console` object. Create a
    new one if it doesn't exist.
    """
    global console
    if console is None:
        console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%m/%d/%Y %H:%M:%S.%f]",
            soft_wrap       = False,
            width           = 200,
            theme           = rich_console_theme,
        )
    return console


def get_error_console() -> rich.console.Console:
    """Get access to the global :class:`rich.console.Console` object that logs
    errors. Create a new one if it doesn't exist.
    """
    global error_console
    if error_console is None:
        error_console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%m/%d/%Y %H:%M:%S.%f]",
            soft_wrap       = False,
            width           = 200,
            stderr          = True,
            style           = "bold red",
            theme           = rich_console_theme,
        )
    return error_console

# endregion


# region Progress

def download_bar() -> progress.Progress:
    """Return a :class:`rich.progress.Progress` object displaying the current
    time, the task description, a progress bar, the percentage complete, the
    transfer speed, the amount downloaded, the time remaining, the time elapsed,
    and a right-pointing arrow.
    """
    return progress.Progress(
        progress.TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
            justify = "left",
            style   = "log.time",
        ),
        progress.TextColumn("{task.description}", justify="right"),
        progress.BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        progress.TransferSpeedColumn(),
        "•",
        progress.DownloadColumn(),
        "•",
        progress.TimeRemainingColumn(),
        ">",
        progress.TimeElapsedColumn(),
        console=console,
    )


def progress_bar() -> progress.Progress:
    """Return a :class:`rich.progress.Progress` object displaying the current
    time, the task description, a progress bar, the percentage complete, the
    total number of processed items, the processing speed, the time remaining,
    the time elapsed, and a spinner.
    """
    return progress.Progress(
        progress.TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
            justify="left", style="log.time"
        ),
        progress.TextColumn("{task.description}", justify="right"),
        progress.BarColumn(bar_width=None, finished_style="green"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        ProcessedItemsColumn(),
        "•",
        ProcessingSpeedColumn(),
        "•",
        progress.TimeRemainingColumn(),
        ">",
        progress.TimeElapsedColumn(),
        progress.SpinnerColumn(),
        console=console,
    )


class GPUMemoryUsageColumn(progress.ProgressColumn):
    """A progress column showing current GPU memory usage, e.g. `33.1/48.0GB`.
    
    Args:
        unit: The unit of memory. Defaults to “GB”.
        table_column: The column in the table to associate this field with.
            Defaults to None.
    """
    
    def __init__(
        self,
        unit        : MemoryUnitType      = constant.MemoryUnit.GB,
        table_column: table.Column | None = None
    ):
        super().__init__(table_column=table_column)
        self.unit = constant.MemoryUnit.from_value(unit)

    def render(self, task: progress.Task) -> text.Text:
        """Return a :class:`rich.text.Text` object showing current GPU memory
        status.
        """
        total, used, free = pynvml.get_gpu_memory()
        memory_status     = f"{used:.1f}/{total:.1f}{self.unit.value}"
        memory_text       = text.Text(memory_status, style="bright_yellow")
        return memory_text
    

class ProcessedItemsColumn(progress.ProgressColumn):
    """A progress column showing number of processed items, e.g. `1728/2025`.
    
    Args:
        table_column: The column in the table to associate this field with.
            Defaults to None.
    """

    def __init__(self, table_column: table.Column | None = None):
        super().__init__(table_column=table_column)

    def render(self, task: progress.Task) -> text.Text:
        """Return a :class:`rich.text.Text` object showing the number of
        processed items.
        """
        completed       = int(task.completed)
        total           = int(task.total)
        download_status = f"{completed}/{total}"
        download_text   = text.Text(download_status, style="progress.download")
        return download_text


class ProcessingSpeedColumn(progress.ProgressColumn):
    """A progress column showing human-readable progressing speed."""
    
    def render(self, task: progress.Task) -> text.Text:
        """Return a :class:`rich.text.Text` object showing the progressing
        speed.
        """
        speed = task.speed
        if speed is None:
            return text.Text("?", style="progress.data.speed")
        speed_data = "{:.2f}".format(speed)
        speed_text = text.Text(f"{speed_data}it/s", style="progress.data.speed")
        return speed_text
        
# endregion


# region Print

def print_dict(data: dict, title: str = ""):
    """Print a dictionary with a title using the :class:`rich.pretty.Pretty`
    format. For example:
    
    Title
    | Key   | Value   |
    |-------|---------|
    | Key 1 | Value 1 |
    | ...   | ...     |
    """
    if isinstance(data, munch.Munch):
        data = data.toDict()
    pr = pretty.Pretty(
        data,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    p  = panel.Panel(pr, title=f"{title}")
    console.log(p)
    

@multipledispatch.dispatch(list)
def print_table(data: list[dict]):
    """Print a list of dictionaries in a :class:`rich.table.Table`. All
    dictionaries in the given list (:param:`data`) must contain the same keys.
    """
    assert isinstance(data, list) and all(isinstance(d, dict) for d in data)
    tab = table.Table(
        show_header  = True,
        header_style = "bold magenta",
    )
    for k, v in data[0].items():
        tab.add_column(k, no_wrap=True)
    for d in data:
        row = [f"{v}" for v in d.values()]
        tab.add_row(*row)
    console.log(table)


@multipledispatch.dispatch(dict)
def print_table(data: dict):
    """Print a dictionary in a :class:`rich.table.Table`."""
    assert isinstance(data, dict)
    tab = table.Table(show_header=True, header_style="bold magenta")
    tab.add_column("Key")
    tab.add_column("Value")
    for k, v in data.items():
        row = [f"{k}", f"{v}"]
        tab.add_row(*row)
    console.log(table)

# endregion
