#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rich Module.

This module extends the :obj:`rich` package.It provides rich text and beautiful
formatting in the terminal, console, and logging throughout the :obj:`mon`
framework.
"""

from __future__ import annotations

__all__ = [
    "GPUMemoryUsageColumn",
    "ProcessedItemsColumn",
    "ProcessingSpeedColumn",
    "console",
    "error_console",
    "field_style",
    "get_console",
    "get_download_bar",
    "get_error_console",
    "get_progress_bar",
    "get_terminal_size",
    "level_styles",
    "print_dict",
    "print_table",
    "rich_console_theme",
    "set_terminal_size",
]

import fcntl
import shutil
import struct
import subprocess
import sys
import termios

import rich
import torch
from plum import dispatch
from rich import panel, pretty, progress, table, text, theme

from mon.core import dtype, utils
from mon.globals import MemoryUnit


# region Console

def get_terminal_size() -> tuple[int, int]:
    """Get the size of the terminal window in columns and rows."""
    size = shutil.get_terminal_size(fallback=(100, 40))
    return size.columns, size.lines


def set_terminal_size(rows: int = 40, cols: int = 100):
    # File descriptor for stdout
    fd   = sys.stdout.fileno()
    # Get terminal size struct
    size = struct.pack("HHHH", rows, cols, 0, 0)
    # Set terminal size
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    subprocess.run(["stty", "rows", str(rows), "cols", str(cols)])


field_style = {
    "asctime"  : {"color": "green"},
    "levelname": {"bold" : True},
    "file_name": {"color": "cyan"},
    "funcName" : {"color": "blue"}
}

level_styles = {
    "critical": {"bold" : True, "color": "red"},
    "debug"   : {"color": "green"},
    "error"   : {"color": "red"},
    "info"    : {"color": "magenta"},
    "warning" : {"color": "yellow"}
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
    log_time_format = "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = True,
    width           = get_terminal_size()[0],  # 150
    theme           = rich_console_theme,
)

error_console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = False,
    width           = get_terminal_size()[0],  # 150
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)


def get_console() -> rich.console.Console:
    """Get access to the global :obj:`rich.console.Console` object. Create a
    new one if it doesn't exist.
    """
    global console
    if console is None:
        console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%m/%d/%Y %H:%M:%S]",
            soft_wrap       = False,
            width           = 150,
            theme           = rich_console_theme,
        )
    return console


def get_error_console() -> rich.console.Console:
    """Get access to the global :obj:`rich.console.Console` object that logs
    errors. Create a new one if it doesn't exist.
    """
    global error_console
    if error_console is None:
        error_console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%m/%d/%Y %H:%M:%S]",
            soft_wrap       = False,
            width           = 150,
            stderr          = True,
            style           = "bold red",
            theme           = rich_console_theme,
        )
    return error_console

# endregion


# region Progress

def get_download_bar(
    transient: bool = False,
    disable  : bool = False,
) -> progress.Progress:
    """Return a :obj:`rich.progress.Progress` object displaying the current
    time, the task description, a progress bar, the percentage complete, the
    transfer speed, the amount downloaded, the time remaining, the time elapsed,
    and a right-pointing arrow.
    """
    return progress.Progress(
        progress.TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S]"),
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
        console   = console,
        transient = transient,
        disable   = disable,
    )


def get_progress_bar(
    transient: bool = False,
    disable  : bool = False,
) -> progress.Progress:
    """Return a :obj:`rich.progress.Progress` object displaying the current
    time, the task description, a progress bar, the percentage complete, the
    total number of processed items, the processing speed, the time remaining,
    the time elapsed, and a spinner.
    """
    return progress.Progress(
        progress.TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S]"),
            justify = "left",
            style   = "log.time"
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
        console   = console,
        transient = transient,
        disable   = disable,
    )


class GPUMemoryUsageColumn(progress.ProgressColumn):
    """A progress column showing current CPU/GPU memory usage,
    e.g. ``33.1/48.0GB``.
    
    Args:
        unit: The unit of memory. Default: ``'GB'``.
        table_column: The column in the table to associate this field with.
            Default: ``None``.
    """
    
    def __init__(
        self,
        devices     : int | list[int] = 0,
        unit        : MemoryUnit      = MemoryUnit.GB,
        table_column: table.Column    = None
    ):
        super().__init__(table_column=table_column)
        self.devices = dtype.to_int_list(devices)
        self.unit    = MemoryUnit.from_value(value=unit)
    
    def render(self, task: progress.Task) -> text.Text:
        """Return a :obj:`rich.text.Text` object showing current GPU memory
        status.
        """
        return self.get_gpu_memory_text(task) \
            if torch.cuda.is_available() \
            else self.get_machine_memory_text(task)
    
    def get_machine_memory_text(self, task: progress.Task) -> text.Text:
        """Return a :obj:`rich.text.Text` object showing current RAM status."""
        total, used, free = utils.get_machine_memory(unit=self.unit)
        memory_status     = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        memory_text       = text.Text(memory_status, style="bright_yellow")
        return memory_text
    
    def get_gpu_memory_text(self, task: progress.Task) -> text.Text:
        """Return a :obj:`rich.text.Text` object showing current GPU memory
        status.
        """
        num_devices = len(self.devices)
        totals, useds, frees = [], [], []
        for i in self.devices:
            t, u, f  = utils.get_gpu_device_memory(device=i, unit=self.unit)
            totals.append(t)
            useds.append(u)
            frees.append(f)
        # total = np.mean(totals)
        # used  = np.mean(useds)
        # free  = np.mean(frees)
        total = min(totals)
        used  = max(useds)
        free  = min(frees)
        memory_status = (f"{used:.1f}/{total:.1f}{self.unit.value} "
                         f"({num_devices} GPUs)")
        memory_text   = text.Text(memory_status, style="bright_yellow")
        return memory_text
    

class ProcessedItemsColumn(progress.ProgressColumn):
    """A progress column showing the number of processed items,
    e.g., ``1728/2025``.
    
    Args:
        table_column: The column in the table to associate this field with.
            Default: ``None``.
    """
    
    def __init__(self, table_column: table.Column = None):
        super().__init__(table_column=table_column)
    
    def render(self, task: progress.Task) -> text.Text:
        """Return a :obj:`rich.text.Text` object showing the number of processed
        items.
        """
        completed     = int(task.completed)
        total         = int(task.total)
        download_text = f"{completed}/{total}"
        download_text = f"{download_text:>14}"
        download_text = text.Text(download_text, style="progress.download")
        return download_text


class ProcessingSpeedColumn(progress.ProgressColumn):
    """A progress column showing human-readable progressing speed."""
    
    def render(self, task: progress.Task) -> text.Text:
        """Return a :obj:`rich.text.Text` object showing the progressing speed.
        """
        speed = task.speed
        if speed is None:
            return text.Text("?", style="progress.data.speed")
        speed_text = "{:0.2f}".format(speed)
        speed_text = f"{speed_text:>7}"
        speed_text = text.Text(f"{speed_text}it/s", style="progress.data.speed")
        return speed_text

# endregion


# region Print

def print_dict(x: dict, title: str = ""):
    """Print a :obj:`dict` with a title using the :obj:`rich.pretty.Pretty`
    format. For example:
    
    Title
    | Key   | Value   |
    |-------|---------|
    | Key 1 | Value 1 |
    | ...   | ...     |
    """
    assert isinstance(x, dict)
    pr = pretty.Pretty(
        x,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    p = panel.Panel(pr, title=f"{title}")
    console.log(p)


@dispatch
def print_table(x: list[dict]):
    """Print a :obj:`list` of :obj:`dict` in a :obj:`rich.table.Table`.
    All :obj:`dict` in the given list must contain the same keys.
    """
    assert isinstance(x, list) and all(isinstance(d, dict) for d in x)
    tab = table.Table(show_header=True, header_style="bold magenta")
    for k, v in x[0].items():
        tab.add_column(k, no_wrap=True)
    for d in x:
        row = [f"{v}" for v in d.values()]
        tab.add_row(*row)
    console.log(tab)


@dispatch
def print_table(x: dict):
    """Print a :obj:`dict` in a :obj:`rich.table.Table`."""
    assert isinstance(x, dict)
    tab = table.Table(show_header=True, header_style="bold magenta")
    tab.add_column("Key")
    tab.add_column("Value")
    for k, v in x.items():
        row = [f"{k}", f"{v}"]
        tab.add_row(*row)
    console.log(tab)

# endregion
