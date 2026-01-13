#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rich-based progress and prompt utility collection.

This module provides helpers to construct progress bars and custom prompt
subclasses.
"""

from __future__ import annotations

__all__ = [
    "create_download_bar",
    "create_progress_bar",
    "MemoryUsageColumn",
    "ProcessedItemsColumn",
    "ProcessingSpeedColumn",
    "SelectionOrInputPrompt",
]

import time
from typing import Any, List, Optional, TextIO

import rich
import torch
from rich.columns import Columns
from rich.console import Console
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
from rich.prompt import DefaultType, InvalidResponse, Prompt, PromptType
from rich.table import Column
from rich.text import Text, TextType

from mon.core.console import console
from mon.core.enum import MemoryUnit
from mon.core.utils import is_int, to_int_list, to_list


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Columns ---

class MemoryUsageColumn(ProgressColumn):
    """Progress column that displays memory usage.

    Display either system RAM or aggregated GPU VRAM usage, depending on whether
    CUDA is available.

    Attributes:
        devices (list[int]): List of GPU device indices to query.
            Defaults to [0].
        unit (MemoryUnit): Memory unit to use for reporting.
            Defaults to MemoryUnit.GB.
        update_interval (float): Minimum time in seconds between updates.
            Defaults to 1.0.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        devices        : int | list[int] = 0,
        unit           : str             = "GB",
        update_interval: float           = 1.0,
        table_column   : Column          = None,
    ):
        """Initialize the memory usage column.

        Args:
            devices: Device index or a list of indices to monitor.
                Defaults to 0.
            unit: Memory unit for display. Defaults to "GB".
            update_interval: Minimum time in seconds between updates.
                Defaults to 1.0.
            table_column: Optional rich.table.Column for custom styling.
                Defaults to None.
        """
        super().__init__(table_column=table_column)
        self.devices         = to_int_list(devices)
        self.unit            = MemoryUnit(value=unit)
        self.update_interval = update_interval
        self._last_update    = 0.0
        self._cached_text    = Text("")

    def render(self, task: Task) -> Text:
        """Render the current memory usage.

        Args:
            task: The rich.progress.Task being rendered.

        Returns:
            Text object displaying the memory usage.
        """
        current_time = time.time()
        if current_time - self._last_update > self.update_interval:
            self._cached_text = self.gpu_memory_text if torch.cuda.is_available() else self.machine_memory_text
            self._last_update = current_time
        return self._cached_text

    @property
    def machine_memory_text(self) -> Text:
        """Format system RAM usage into a Text object."""
        # Import locally to avoid circular dependencies.
        from mon.core.device import query_ram_usages

        total, used, _ = query_ram_usages(unit=self.unit)
        memory_status  = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        return Text(memory_status, style="bright_yellow")

    @property
    def gpu_memory_text(self) -> Text:
        """Format and aggregate GPU VRAM usage into a Text object."""
        # Import locally to avoid circular dependencies.
        from mon.core.device import query_vram_usage

        num_devices = len(self.devices)
        total_mem, used_mem = 0.0, 0.0
        for i in self.devices:
            total, used, _ = query_vram_usage(device=i, unit=self.unit)
            total_mem += total
            used_mem  += used

        memory_status = f"{used_mem:.1f}/{total_mem:.1f}{self.unit.value} ({num_devices} GPUs)"
        return Text(memory_status, style="bright_yellow")


class ProcessedItemsColumn(ProgressColumn):
    """Progress column that displays the count of processed items.

    Show a \"completed/total\" count in a fixed-width field to prevent the
    progress bar from resizing during updates.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, table_column: Column = None):
        """Initialize a new instance.

        Args:
            table_column: Optional rich.table.Column for custom styling.
                Defaults to None.
        """
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        """Render the processed items count for the task.

        Args:
            task: The rich.progress.Task being rendered.

        Returns:
            Text object showing \"completed/total\" items.
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

    def render(self, task: Task) -> Text:
        """Render the processing speed for the task.

        Args:
            task: The rich.progress.Task being rendered.

        Returns:
            Text object displaying the processing speed.
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


# --- Prompts ---

class SelectionOrInputPrompt(Prompt):
    """Prompt that supports selection by index, direct value, or free-form input.

    Extend ``rich.prompt.Prompt`` to create a flexible prompt that can present a
    list of choices for selection while also accepting arbitrary input if it
    doesn't match a choice.

    Attributes:
        allow_empty (bool): If True, allow an empty string as a valid response.
            Defaults to False.
        column_first (bool): If True, print choices in column-first order.
            Defaults to False.
    """

    response_type: type = str

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        prompt        : TextType            = "",
        *,
        console       : Optional[Console]   = None,
        password      : bool                = False,
        choices       : Optional[List[str]] = None,
        case_sensitive: bool                = True,
        show_default  : bool                = True,
        show_choices  : bool                = True,
        column_first  : bool                = False,
        allow_empty   : bool                = False,
    ):
        """Initialize the prompt.

        Args:
            prompt: Text to display for the prompt.
            console: Optional Console object. Defaults to None.
            password: If True, hide input. Defaults to False.
            choices: List of choices to display for selection. Defaults to None.
            case_sensitive: If True, choice matching is case-sensitive.
                Defaults to True.
            show_default: If True, display the default value. Defaults to True.
            show_choices: If True, display the list of choices.
                Defaults to True.
            column_first: If True, print choices in column-first order.
                Defaults to False.
            allow_empty: If True, allow an empty string as a valid response.
                Defaults to False.
        """
        self.allow_empty  = allow_empty
        self.column_first = column_first
        super().__init__(
            prompt,
            console        = console,
            password       = password,
            choices        = choices,
            case_sensitive = case_sensitive,
            show_default   = show_default,
            show_choices   = show_choices,
        )

    def print_choices(self):
        """Print available choices in columns."""
        choices_ = []
        for i, choice in enumerate(self.choices):
            choices_.append(f"{f'{i}.':>6} {choice}")
        columns = Columns(choices_, equal=True, column_first=self.column_first)
        rich.print(columns)

    @classmethod
    def ask(
        cls,
        prompt        : TextType            = "",
        *,
        console       : Optional[Console]   = None,
        password      : bool                = False,
        choices       : Optional[List[str]] = None,
        case_sensitive: bool                = True,
        show_default  : bool                = True,
        show_choices  : bool                = True,
        allow_empty   : bool                = False,
        column_first  : bool                = False,
        default       : Any                 = ...,
        stream        : Optional[TextIO]    = None,
    ) -> Any:
        """Create and run a SelectionOrInputPrompt instance.

        Args:
            prompt: Text to display for the prompt. Defaults to \"\".
            console: Optional Console object. Defaults to None.
            password: If True, hide input. Defaults to False.
            choices: List of choices to display for selection. Defaults to None.
            case_sensitive: If True, choice matching is case-sensitive.
                Defaults to True.
            show_default: If True, display the default value. Defaults to True.
            show_choices: If True, display the list of choices.
                Defaults to True.
            allow_empty: If True, allow an empty string as a valid response.
                Defaults to False.
            column_first: If True, print choices in column-first order.
                Defaults to False.
            default: Default value to use if the user enters nothing.
                Defaults to ....
            stream: Optional text stream for I/O. Defaults to None.

        Returns:
            Processed user response.
        """
        _prompt = cls(
            prompt,
            console        = console,
            password       = password,
            choices        = choices,
            case_sensitive = case_sensitive,
            show_default   = show_default,
            show_choices   = show_choices,
            allow_empty    = allow_empty,
            column_first   = column_first,
        )
        return _prompt(default=default, stream=stream)

    def render_default(self, default: DefaultType) -> Text:
        """Render the default value for display.

        Args:
            default: Default value to render.

        Returns:
            Text instance that shows the provided default.
        """
        return Text(f"[{default}]", "prompt.default")

    def make_prompt(self, default: DefaultType) -> Text:
        """Build the prompt text to display.

        Args:
            default: Default value to show in the prompt when provided.

        Returns:
            Constructed prompt Text instance.
        """
        if self.show_choices and self.choices and len(self.choices) > 0:
            rich.print(self.prompt)
            self.print_choices()
            prompt = Text.from_markup("", style="prompt")
        else:
            prompt = self.prompt.copy()
        prompt.end = ""

        if (
            default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt

    def check_choice(self, value: str) -> bool:
        """Validate that a value is among the valid choices.

        Args:
            value: Candidate value to validate.

        Returns:
            True when the provided value matches one of the configured choices.
        """
        assert self.choices is not None
        if self.case_sensitive:
            return value in self.choices
        return value.lower() in [choice.lower() for choice in self.choices]

    def process_response(self, value: str) -> PromptType:
        """Validate and convert the user's response.

        Args:
            value: Raw user input string.

        Returns:
            Validated and possibly converted response.

        Raises:
            InvalidResponse: When the provided ``value`` is not acceptable.
        """
        value = value.strip() if isinstance(value, str) else value

        if self.choices:
            if not value and not self.allow_empty:
                raise InvalidResponse(self.illegal_choice_message)

            # Split input to support multi-index/multi-value selection (e.g., "0,2")
            input_parts      = to_list(value, sep=[",", ";"])
            processed_values = []

            for part in input_parts:
                part = part.strip()
                # Check if part is a valid index
                if is_int(part):
                    idx = int(part)
                    if 0 <= idx < len(self.choices):
                        processed_values.append(self.choices[idx])
                    else:
                        raise IndexError(f"Index {idx} out of range for choices "
                                         f"of size {len(self.choices)}.")
                # Check if part is a direct choice match
                elif self.check_choice(part):
                    processed_values.append(part)
                # Handle free-form input (if allowed) or error
                else:
                    processed_values.append(part)

            # Return single value if only one was selected, else the list
            return processed_values[0] if len(processed_values) == 1 else processed_values

        return value

    # --- Callable & Context Manager ---
    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Prompt until a valid response is obtained.

        Args:
            default: Default value to use when input is empty.
                Defaults to ....
            stream: Optional text stream for input/output. Defaults to None.

        Returns:
            Processed user response.
        """
        while True:
            self.pre_prompt()
            prompt = self.make_prompt(default)
            value  = self.get_input(self.console, prompt, self.password, stream=stream)
            if value == "" and default != ...:
                # return default
                value = default
            try:
                return_value = self.process_response(value)
            except (InvalidResponse, IndexError) as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_download_bar(transient: bool = False, disable: bool = False) -> Progress:
    """Create a download progress bar.

    Args:
        transient: If True, remove the progress display after completion.
            Defaults to False.
        disable: If True, disable the progress display entirely.
            Defaults to False.

    Returns:
        Configured Progress instance for download tasks.
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
    transient  : bool = False,
    disable    : bool = False,
    show_memory: bool = True
) -> Progress:
    """Create a general-purpose progress bar for tasks.

    Args:
        transient: If True, remove the progress display after completion.
            Defaults to False.
        disable: If True, disable the progress display entirely.
            Defaults to False.
        show_memory: If True, include a column for memory usage.
            Defaults to True.

    Returns:
        Configured Progress instance for general tasks.
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
