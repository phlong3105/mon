#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Progress Bars.

This module extends the ``rich.progress`` module with custom progress bars and
download bars.
"""

from __future__ import annotations

__all__ = [
    "create_download_bar",
    "create_progress_bar",
    "MemoryUsageColumn",
    "ProcessedItemsColumn",
    "ProcessingSpeedColumn",
]

import time
from typing import override

import torch
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column
from rich.text import Text

from mon.core.console import console
from mon.core.context import sys_ctx
from mon.core.dtype import MemoryUnit
from mon.core.typing import MemoryUnitLike


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

# --- Columns ---

class MemoryUsageColumn(ProgressColumn):
    """Progress column that displays memory usage.

    Display either system RAM or aggregated GPU VRAM usage, depending on whether
    CUDA is available.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        unit: MemoryUnitLike = "GB",
        update_interval: float = 1.0,
        table_column: Column | None = None,
    ):
        """Initialize a new instance.

        Args:
            unit (MemoryUnitType, optional): Memory unit to use for reporting.
                Defaults to "GB".
            update_interval (float, optional): Minimum time in seconds between
                updates. Defaults to 1.0.
            table_column (Column, optional): Column for custom styling.
                Defaults to None.
        """
        super().__init__(table_column=table_column)
        self.unit = MemoryUnit(unit)
        self.update_interval = update_interval
        self.last_update = 0.0
        self.cached_text = Text("")

    # --- Properties ---
    @property
    def machine_memory_text(self) -> Text:
        """Return formatted system RAM usage as a Text object."""
        cpu = sys_ctx.cpu
        total, used, _ = cpu.usages(self.unit)
        memory_status = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        return Text(memory_status, style="bright_yellow")

    @property
    def gpu_memory_text(self) -> Text:
        """Return formatted GPU VRAM usage as a Text object."""
        cudas = sys_ctx.cudas
        total_mem, used_mem = 0.0, 0.0
        for d in cudas:
            total, used, _ = d.usages(self.unit)
            total_mem += total
            used_mem += used

        memory_status = f"{used_mem:.1f}/{total_mem:.1f}{self.unit.value} ({len(cudas)} GPUs)"
        return Text(memory_status, style="bright_yellow")

    # --- Callable & Context Manager ---
    @override
    def render(self, task: Task) -> Text:
        """Render the current memory usage.

        Args:
            task (Task): The rich.progress.Task being rendered.

        Returns:
            Text: Memory usage formatted for display.
        """
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self.cached_text = self.machine_memory_text
            if torch.cuda.is_available():
                self.cached_text += self.gpu_memory_text
            self.last_update = current_time
        return self.cached_text


class ProcessedItemsColumn(ProgressColumn):
    r"""Progress column that displays the count of processed items.

    Show a \"completed/total\" count in a fixed-width field to prevent the
    progress bar from resizing during updates.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, table_column: Column = None):
        """Initialize a new instance.

        Args:
            table_column (Column, optional): Optional rich.table.Column for
                custom styling. Defaults to None.
        """
        super().__init__(table_column=table_column)

    # --- Callable & Context Manager ---
    @override
    def render(self, task: Task) -> Text:
        r"""Render the processed items count for the task.

        Args:
            task (Task): The rich.progress.Task being rendered.

        Returns:
            Text: Processed items count formatted for display.
        """
        completed = int(task.completed)

        if task.total is None or task.total == float("inf"):
            # Handle cases where the total is unknown (e.g., streaming).
            count = f"{completed}"
        else:
            total = int(task.total)
            count = f"{completed}/{total}"

        # Use a fixed width to prevent the progress bar from "jumping" as numbers grow.
        return Text(f"{count:>14}", style="progress.download")


class ProcessingSpeedColumn(ProgressColumn):
    """Progress column that displays processing speed.

    Show the speed in items per second (it/s) or, if the speed is less than 1
    it/s, show the latency in milliseconds per item (ms/it).
    """

    # --- Callable & Context Manager ---
    @override
    def render(self, task: Task) -> Text:
        """Render the processing speed for the task.

        Args:
            task (Task): The rich.progress.Task being rendered.

        Returns:
            Text: Processing speed formatted for display.
        """
        speed = task.speed
        if speed is None or speed == 0:
            return Text("?", style="progress.data.speed")

        # If speed is slow, it's more intuitive to show latency.
        if speed < 1.0:
            latency_ms = (1.0 / speed) * 1000
            speed_text = f"{latency_ms:>.1f}ms/it"
        else:
            speed_text = f"{speed:>.2f}it/s"

        return Text(f"{speed_text:>10}", style="progress.data.speed")

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_download_bar(
    transient: bool = False,
    disable: bool = False,
) -> Progress:
    """Create a download progress bar.

    Args:
        transient (bool, optional): If True, remove the progress display after
            completion. Defaults to False.
        disable (bool, optional): If True, disable the progress display entirely.
            Defaults to False.

    Returns:
        Progress: Configured Progress instance for download tasks.
    """
    columns = [
        TextColumn(console.get_datetime().strftime("[%X]"), justify="left", style="log.time"),
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
    ]
    return Progress(*columns, console=console, transient=transient, disable=disable)


def create_progress_bar(
    transient: bool = False,
    disable: bool = False,
    show_memory: bool = False,
) -> Progress:
    """Create a general-purpose progress bar for tasks.

    Args:
        transient (bool, optional): If True, remove the progress display after
            completion. Defaults to False.
        disable (bool, optional): If True, disable the progress display entirely.
            Defaults to False.
        show_memory (bool, optional): If True, show memory usage in the progress
            bar. Defaults to False.

    Returns:
        Progress: Configured Progress instance for general tasks.
    """
    columns = [
        TextColumn(console.get_datetime().strftime("[%X]"), justify="left", style="log.time"),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None, finished_style="green"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        ProcessedItemsColumn(),
        "•",
        ProcessingSpeedColumn(),
    ]
    if show_memory:
        columns.extend(["•", MemoryUsageColumn()])
    columns.extend(["•", TimeRemainingColumn(), ">", TimeElapsedColumn(), SpinnerColumn()])

    return Progress(*columns, console=console, transient=transient, disable=disable)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
