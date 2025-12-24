#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rich-based progress and prompt utility collection.

This module provides helpers to construct Progress instances and custom
ProgressColumn and prompt subclasses for interactive command-line flows,
including download and general-purpose progress bars, memory and processing
columns, and a prompt class that accepts either selection indices or free-form
input.
"""

__all__ = [
    "create_download_bar",
    "create_progress_bar",
]

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
# PROGRESS TRACKING SYSTEMS
# ==============================================================================

# --- Specialized Bars (Download vs. General Task configurations) ---
def create_download_bar(transient: bool = False, disable: bool = False) -> Progress:
    """Create a download progress bar.

    Args:
        transient: If True, remove progress display after completion.
        disable: If True, disable the progress display.

    Returns:
        Configured Progress instance for download tasks.
    """
    return Progress(
        TextColumn(
            # console.get_datetime().strftime("[%x %H:%M:%S]"),
            console.get_datetime().strftime("[%X]"),
            justify="left",
            style="log.time",
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
        console   = console,
        transient = transient,
        disable   = disable,
    )


def create_progress_bar(transient: bool = False, disable: bool = False) -> Progress:
    """Create a general-purpose progress bar.

    Args:
        transient: If True, remove progress display after completion.
        disable: If True, disable the progress display.

    Returns:
        Configured Progress instance for general tasks.
    """
    return Progress(
        TextColumn(
            # console.get_datetime().strftime("[%x %H:%M:%S]"),
            console.get_datetime().strftime("[%X]"),
            justify="left",
            style="log.time"
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
        console   = console,
        transient = transient,
        disable   = disable,
    )


# --- Telemetry Columns (Memory, Processing Speed, and Item counters) ---
class MemoryUsageColumn(ProgressColumn):
    """A progress column for memory usage.

    Display machine or GPU memory usage aggregated across configured devices.

    Attributes:
        devices (list[int]): GPU device indices to query.
        unit (MemoryUnit): Unit used for reporting values.
    """

    def __init__(
        self,
        devices     : int | list[int] = 0,
        unit        : str    = "GB",
        table_column: Column = None
    ):
        """Initialize a new instance.

        Configure which GPU device(s) to query and the memory unit to display.

        Args:
            devices: Device index or list of device indices.
            unit: Unit label used for display.
            table_column: Optional associated table column.
        """
        super().__init__(table_column=table_column)
        self.devices = to_int_list(devices)
        self.unit    = MemoryUnit(value=unit)
    
    def render(self, task: Task) -> Text:
        """Render current memory usage for the task.

        Args:
            task: Rich Task representing the progress.

        Returns:
            Text instance describing the memory usage (CPU or GPU).
        """
        return self.gpu_memory_text \
            if torch.cuda.is_available() \
            else self.machine_memory_text
    
    @property
    def machine_memory_text(self) -> Text:
        """Format machine (RAM) memory usage.

        Returns:
            Text instance with CPU memory usage summary.
        """
        from mon.core.device import query_ram_usages
        
        total, used, _ = query_ram_usages(unit=self.unit)
        memory_status  = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        memory_text    = Text(memory_status, style="bright_yellow")
        return memory_text
    
    @property
    def gpu_memory_text(self) -> Text:
        """Format aggregated GPU memory usage across devices.

        Returns:
            Text instance with aggregated GPU memory usage summary.
        """
        from mon.core.device import query_vram_usage
        
        num_devices = len(self.devices)
        totals, useds = [], []
        for i in self.devices:
            total, used, _ = query_vram_usage(device=i, unit=self.unit)
            totals.append(total)
            useds.append(used)
        total = min(totals)
        used  = max(useds)
        memory_status = f"{used:.1f}/{total:.1f}{self.unit.value} ({num_devices} GPUs)"
        memory_text   = Text(memory_status, style="bright_yellow")
        return memory_text


class ProcessedItemsColumn(ProgressColumn):
    """A progress column showing processed item counts.

    Present completed/total counts in a fixed-width field.
    """

    def __init__(self, table_column: Column = None):
        """Initialize a new instance.

        Args:
            table_column: Optional associated table column.
        """
        super().__init__(table_column=table_column)
    
    def render(self, task: Task) -> Text:
        """Render processed items count for the task.

        Args:
            task: Rich Task representing the progress.

        Returns:
            Text instance showing completed/total items for the task.
        """
        completed = int(task.completed)
        total     = int(task.total)
        count     = f"{completed}/{total}"
        count     = f"{count:>14}"
        return Text(count, style="progress.download")


class ProcessingSpeedColumn(ProgressColumn):
    """A progress column showing processing speed.

    Show task processing speed in items per second or a placeholder when
    unknown.
    """

    def render(self, task: Task) -> Text:
        """Render processing speed for the task.

        Args:
            task: Rich Task representing the progress.

        Returns:
            Text instance displaying the processing speed or a placeholder.
        """
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        speed_text = f"{speed:0.2f}"
        speed_text = f"{speed_text:>7}"
        return Text(f"{speed_text}it/s", style="progress.data.speed")


# ==============================================================================
# INTERACTIVE CLI FLOWS
# ==============================================================================

# --- Custom Prompts ---
class SelectionOrInputPrompt(Prompt):
    """A selection-or-input prompt.

    Support selection by index or direct input, optional empty input, and
    configurable choice display.

    Attributes:
        response_type (type): Expected response type for default rendering.
        allow_empty (bool): Whether empty input is permitted.
        column_first (bool): Whether columns are printed column-first.
        choices (Optional[list[str]]): Optional list of choices for selection.
        password (bool): Whether the prompt is a password prompt.
    """

    response_type: type = str

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
        """Initialize a new instance.

        Args:
            prompt: Prompt text to display.
            console: Optional Console to render the prompt.
            password: If True, hide input.
            choices: Optional list of choices to present.
            case_sensitive: Whether choice matching is case sensitive.
            show_default: Whether to display the default value.
            show_choices: Whether to display choices.
            column_first: Whether to print choices column-first.
            allow_empty: Whether to accept empty input.
        """
        self.allow_empty  = allow_empty
        self.column_first = column_first
        super().__init__(
            prompt         = prompt,
            console        = console,
            password       = password,
            choices        = choices,
            case_sensitive = case_sensitive,
            show_default   = show_default,
            show_choices   = show_choices,
        )

    def print_choices(self):
        """Print available choices in columns.

        Print the available choices as aligned columns to the console.
        """
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
        """Create and run a selection-or-input prompt.

        Args:
            prompt: Prompt text to display.
            console: Optional Console to render the prompt.
            password: If True, hide input.
            choices: Optional list of choices to present.
            case_sensitive: Whether choice matching is case sensitive.
            show_default: Whether to display the default value.
            show_choices: Whether to display choices.
            allow_empty: Whether to accept empty input.
            column_first: Whether to print choices column-first.
            default: Default value to use when input is empty.
            stream: Optional text stream for input/output.

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

        Accept indices or values, convert indices to choices, and raise an
        InvalidResponse for invalid input.

        Args:
            value: Raw user input string.

        Returns:
            Validated and possibly converted response.

        Raises:
            InvalidResponse: When the provided ``value`` is not acceptable.
        """
        value = value.strip() if isinstance(value, str) else value

        if self.choices is not None:
            if len(self.choices) == 0:
                return value
            if len(self.choices) > 0 and value == "" and not self.allow_empty:
                raise InvalidResponse(self.illegal_choice_message)
            # If the whole value is a choice, return it
            if value in self.choices:
                return value

            # Convert index (if any) to choice
            value = to_list(value, sep=[",", ";"])
            if any(v for v in value if is_int(v) and not 0 <= int(v) <= len(self.choices) - 1):
                raise InvalidResponse(self.illegal_choice_message)
            value = [self.choices[int(v)] if is_int(v) else v for v in value]
            
        return value
    
    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Prompt until a valid response is obtained.

        Loop until the response validates, then return the processed value.

        Args:
            default: Default value to use when input is empty.
            stream: Optional text stream for input/output.

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
            except InvalidResponse as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value
