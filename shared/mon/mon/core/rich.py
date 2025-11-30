#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for rich progress bars and prompts.

This module implements utilities for creating rich progress bars and prompts
using the ``rich`` library. It includes functions to create download and general
progress bars, as well as a custom prompt class that allows for selection or
direct input.
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


# ----- Progress -----
def create_download_bar(transient: bool = False, disable: bool = False) -> Progress:
    """Creates a rich.progress.Progress for download tracking.

    Args:
        transient (bool): If true, the progress bar will only display transient
            progress. Defaults to False.
        disable (bool): If true, the progress bar will be disabled. Defaults to
            False.

    Returns:
        rich.progress.Progress: A progress bar with download-specific columns.
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
    """Creates a rich.progress.Progress for general progress tracking.

    Args:
        transient (bool): If true, the progress bar will only display transient
            progress. Defaults to False.
        disable (bool): If true, the progress bar will be disabled. Defaults to
            False.

    Returns:
        rich.progress.Progress: A progress bar with general-purpose columns.
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


class MemoryUsageColumn(ProgressColumn):
    """Displays CPU/GPU memory usage in a progress bar (e.g., 33.1/48.0GB)."""
    
    def __init__(
        self,
        devices     : int | list[int] = 0,
        unit        : str    = "GB",
        table_column: Column = None
    ):
        """Initializes the MemoryUsageColumn.
        
        Args:
            devices (int): GPU device index or list of indices. Defaults to 0.
            unit (str): Memory unit (e.g., 'GB'). Defaults to 'GB'.
            table_column (Column, optional): Column in table to associate with.
                Defaults to None.
        """
        super().__init__(table_column=table_column)
        self.devices = to_int_list(devices)
        self.unit    = MemoryUnit(value=unit)
    
    def render(self, task: Task) -> Text:
        """Renders current GPU or CPU memory usage as text.

        Args:
            task: rich.progress.Task object for the progress task.

        Returns:
            rich.text.Text with memory usage status.
        """
        return self.gpu_memory_text \
            if torch.cuda.is_available() \
            else self.machine_memory_text
    
    @property
    def machine_memory_text(self) -> Text:
        """Renders current RAM usage as text.

        Returns:
            rich.text.Text with RAM usage status.
        """
        from mon.core.device import get_memory_usages
        
        total, used, _ = get_memory_usages(unit=self.unit)
        memory_status  = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        memory_text    = Text(memory_status, style="bright_yellow")
        return memory_text
    
    @property
    def gpu_memory_text(self) -> Text:
        """Renders current GPU memory usage as text.

        Returns:
            rich.text.Text with GPU memory usage status.
        """
        from mon.core.device import get_cuda_memory_usages
        
        num_devices = len(self.devices)
        totals, useds = [], []
        for i in self.devices:
            total, used, _ = get_cuda_memory_usages(device=i, unit=self.unit)
            totals.append(total)
            useds.append(used)
        total = min(totals)
        used  = max(useds)
        memory_status = f"{used:.1f}/{total:.1f}{self.unit.value} ({num_devices} GPUs)"
        memory_text   = Text(memory_status, style="bright_yellow")
        return memory_text


class ProcessedItemsColumn(ProgressColumn):
    """Shows number of processed items in a progress bar (e.g., 1728/2025)."""
    
    def __init__(self, table_column: Column = None):
        """Initializes the ProcessedItemsColumn.
        
        Args:
            table_column (Column, optional): Column in table to associate with.
                Defaults to None.
        """
        super().__init__(table_column=table_column)
    
    def render(self, task: Task) -> Text:
        """Renders the number of processed items as text.

        Args:
            task: rich.progress.Task object for the progress task.

        Returns:
            rich.text.Text with processed items count.
        """
        completed = int(task.completed)
        total     = int(task.total)
        count     = f"{completed}/{total}"
        count     = f"{count:>14}"
        return Text(count, style="progress.download")


class ProcessingSpeedColumn(ProgressColumn):
    """Shows human-readable processing speed in a progress bar."""
    
    def render(self, task: Task) -> Text:
        """Renders the processing speed as text.

        Args:
            task: rich.progress.Task object for the progress task.

        Returns:
            rich.text.Text with the processing speed.
        """
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        speed_text = f"{speed:0.2f}"
        speed_text = f"{speed_text:>7}"
        return Text(f"{speed_text}it/s", style="progress.data.speed")


# ----- Prompt Class -----
class SelectionOrInputPrompt(Prompt):
    """Extends rich.prompt.Prompt to allow for either selecting an index or
    directly entering value.
    
    Attributes:
        response_type (type): The expected type of the response value.
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
        """Initializes the SelectionOrInputPrompt.
        
        Args:
            prompt (TextType): Prompt text. Defaults to "".
            console (Console, optional): Console instance or None to use global
                console. Defaults to None.
            password (bool): Enable password input. Defaults to False.
            choices (list[str]): List of column names. Defaults to None.
            case_sensitive (bool): Matching of choices should be case-sensitive.
                Defaults to True.
            show_default (bool): Show default in prompt. Defaults to True.
            show_choices (bool): Show choices in prompt. Defaults to True.
            allow_empty (bool): Allow empty input. Defaults to False.
            column_first (bool): Align items from top to bottom (rather
                than left to right). Defaults to False.
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
        """Prints columns of choices to the console."""
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
        """Shortcuts to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt (TextType): Prompt text. Defaults to "".
            console (Console, optional): Console instance or None to use global
                console. Defaults to None.
            password (bool): Enable password input. Defaults to False.
            choices (list[str]): List of column names. Defaults to None.
            case_sensitive (bool): Matching of choices should be case-sensitive.
                Defaults to True.
            show_default (bool): Show default in prompt. Defaults to True.
            show_choices (bool): Show choices in prompt. Defaults to True.
            allow_empty (bool): Allow empty input. Defaults to False.
            column_first (bool): Align items from top to bottom (rather
                than left to right). Defaults to False.
            default (Any, optional): Optional default value.
            stream (TextIO, optional): Stream to read input from. Defaults to
                None.
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
        """Turns the supplied default in to a Text instance.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text containing rendering of default value.
        """
        return Text(f"[{default}]", "prompt.default")
    
    def make_prompt(self, default: DefaultType) -> Text:
        """Makes prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text to display in prompt.
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
        """Checks value is in the list of valid choices.

        Args:
            value (str): Value entered by user.

        Returns:
            bool: True if value is a valid choice, False otherwise.
        """
        assert self.choices is not None
        if self.case_sensitive:
            return value in self.choices
        return value.lower() in [choice.lower() for choice in self.choices]
    
    def process_response(self, value: str) -> PromptType:
        """Processes response from user, convert to prompt type.

        Args:
            value (str): String typed by user.

        Raises:
            If ``value`` is invalid.

        Returns:
            PromptType: Processed value.
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
            
            '''
            for i, v in enumerate(value):
                if not self.check_choice(v):
                    raise InvalidResponse(self.illegal_choice_message)
                if not self.case_sensitive:
                    # return the original choice, not the lower case version
                    value[i] = self.choices[[choice.lower() for choice in self.choices].index(v.lower())]
            '''
            # value = value[0] if len(value) == 1 else value
            
        return value
    
    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Runs the prompt loop.

        Args:
            default (Any, optional): Optional default value.

        Returns:
            PromptType: Processed value.
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
