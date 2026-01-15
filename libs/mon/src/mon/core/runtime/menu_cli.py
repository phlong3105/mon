#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Interactive CLI menu utilities.

This module provides prompt wrappers and interactive flows to collect runtime
options using Rich-based prompts.
"""

from __future__ import annotations

__all__ = [
    "RunCLI",
]

from typing import Any

import box
from rich import prompt

from mon.core.console import console, rprint_dict
from mon.core.pathlib import Path
from mon.core.rich import SelectionOrInputPrompt
from mon.core.utils import is_int, to_int, to_list, to_str
from .core import CLI_OPTIONS, DEFAULT_ARGS
from .resolve import (
    load_config,
    parse_model_dir,
    parse_weights_file,
    ProjectResolver,
)


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class Prompt:
    """Wrapper for interactive selection or input prompts.

    Store prompt text, default, choices, and the last returned value. Normalize
    and display prompts using the Rich-based ``SelectionOrInputPrompt``.

    Attributes:
        text (str): Prompt text.
        default (str): Normalized default string.
        choices (list[str] | None): Normalized choices list or None.
        value (str): Last returned value from ``prompt()``.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        text   : str,
        default: str,
        choices: list[str] | None = None
    ):
        """Initialize a new instance.

        Args:
            text: Prompt text to display.
            default: Default value to show.
            choices: Optional sequence of choices for selection prompts.
                Defaults to None.
        """
        self.text    = text
        self.default = default
        self.choices = choices
        self.value   = None

    # --- Properties ---
    @property
    def default(self) -> str:
        """Return the default as a displayable string."""
        return self._default

    @default.setter
    def default(self, default: str):
        """Set and normalize the default value.

        Args:
            default: Default value to set.
        """
        self._default = str(default) if default else ""

    @property
    def value(self) -> str:
        """Return the last stored value."""
        return self._value

    @value.setter
    def value(self, value: str):
        """Normalize and store a value returned from the prompt.

        Args:
            value: Value to store.
        """
        if value:
            value = value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value
        else:
            value = ""
        self._value = value

    @property
    def choices(self) -> list[str] | None:
        """Return normalized choices for display."""
        return self._choices

    @choices.setter
    def choices(self, value: list[str] | None = None):
        """Normalize and store choices.

        Args:
            value: Choices to store. Defaults to None.
        """
        self._choices = to_list(value) if value else None

    # --- Callable & Context Manager ---
    def prompt(self) -> Any:
        """Display the prompt and return the user's response.

        Returns:
            User's response.
        """
        kwargs = {
            "prompt"        : self.text,
            "choices"       : self.choices,
            "case_sensitive": True,
            "show_default"  : True,
            "show_choices"  : True,
            "allow_empty"   : False,
            "column_first"  : False,
            "default"       : self.default,
        }
        self.value = SelectionOrInputPrompt().ask(**kwargs)
        return self.value


class Confirm:
    """Boolean confirmation prompt wrapper.

    Store prompt text, default boolean, and last returned value. Display a
    confirmation prompt using Rich.

    Attributes:
        text (str): Prompt text.
        default (bool): Default boolean selection.
        value (bool): Last returned value from ``prompt()``.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, text: str, default: bool = True):
        """Initialize a new instance.

        Args:
            text: Prompt text to display.
            default: Default boolean selection. Defaults to True.
        """
        self.text    = text
        self.default = default
        self.value   = default

    # --- Callable & Context Manager ---
    def prompt(self) -> bool:
        """Ask for confirmation and return the result.

        Returns:
            Confirmation result.
        """
        self.value = prompt.Confirm().ask(prompt=self.text, default=self.default)
        return self.value


class NumberPrompt:
    """Integer prompt wrapper for numeric input.

    Store prompt text, default integer, and last returned value. Normalize
    numeric input and display an integer prompt using Rich.

    Attributes:
        text (str): Prompt text.
        default (int): Default numeric value or -1 for unset.
        value (int | None): Last returned value from ``prompt()``.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, text: str, default: int = -1):
        """Initialize a new instance.

        Args:
            text: Prompt text.
            default: Default numeric value or -1 for unset. Defaults to -1.
        """
        self.text    = text
        self.default = default
        self.value   = default

    # --- Properties ---
    @property
    def default(self):
        """Return the normalized numeric default."""
        return self._default

    @default.setter
    def default(self, value: int):
        """Normalize and set the numeric default.

        Args:
            value: Default value to set.
        """
        value       = value[0] if isinstance(value, (list, tuple)) else value
        value       = to_int(value)
        self._default = value if isinstance(value, (int, float)) else -1

    @property
    def value(self) -> int:
        """Return the stored numeric value."""
        return self._value

    @value.setter
    def value(self, value: int):
        """Normalize and set the numeric value.

        Args:
            value: Value to set.
        """
        value       = value[0] if isinstance(value, (list, tuple)) else value
        value       = to_int(value)
        self._value = None if isinstance(value, (int, float)) and value < 0 else value

    # --- Callable & Context Manager ---
    def prompt(self) -> int:
        """Prompt for an integer and return the normalized value.

        Returns:
            Normalized integer value.
        """
        self.value = prompt.IntPrompt().ask(prompt=self.text, default=self.default)
        return self.value


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class WeightsPrompt(Prompt):
    """Weights selection prompt.

    Provide a prompt for selecting weights files from available options.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        text   : str = "",
        default: str | Path | None = "",
        choices: list[str] | list[Path] | None = None,
    ):
        """Initialize a new instance.

        Args:
            text: Prompt text.
            default: Default weights.
            choices: Optional override list of choices. Defaults to None.
        """
        default = str(default) if default else None
        choices = [str(c) for c in choices]
        super().__init__(text=text, default=default, choices=choices)

    # --- Properties ---
    @property
    def value(self):
        """Return the normalized weights selection."""
        return self._value

    @value.setter
    def value(self, value: Any):
        """Normalize and set chosen weights.

        Args:
            value: Value to set.
        """
        value = value if value not in [None, ""] else None
        if value:
            if isinstance(value, str):
                value = to_list(value)
            if self.choices and len(self.choices) > 0:
                value = [self.choices[int(w)] if is_int(w) else w for w in value]
                value = [w.replace("'", "") for w in value]

            value = value[0] if isinstance(value, (list, tuple)) else value
            # TODO: Delete later
            # value = value[0] if len(value) == 1 else value

        self._value = value

    # --- Callable & Context Manager ---
    def prompt(self) -> Any:
        """Display weights prompt allowing empty input.

        Returns:
            User's response.
        """
        kwargs = {
            "prompt"        : self.text,
            "choices"       : self.choices,
            "case_sensitive": True,
            "show_default"  : True,
            "show_choices"  : True,
            "allow_empty"   : True,
            "default"       : self.default,
        }
        self.value = SelectionOrInputPrompt().ask(**kwargs)
        return self.value


class DataPrompt(Prompt):
    """Dataset selection prompt.

    Provide a prompt for selecting datasets from available options.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        text   : str = "",
        default: str = "",
        choices: list[str] | None = None,
    ):
        """Initialize a new instance.

        Args:
            text: Prompt text.
            default: Default data.
            choices: Optional override list of choices. Defaults to None.
        """
        default = to_str(default, sep=", ")
        # default = wrap_str(default, max_length=get_terminal_size()[0])
        super().__init__(text=text, default=default, choices=choices)

    # --- Properties ---
    @property
    def value(self) -> list[str]:
        """Return the normalized data selection as a list."""
        return self._value

    @value.setter
    def value(self, value: str):
        """Normalize and store the data selection.

        Args:
            value: Value to set.
        """
        if value:
            value = to_list(value)
        else:
            value = []
        self._value = value

# endregion


# ==============================================================================
# region CONTROL
# ==============================================================================

class RunCLI:
    """Interactive runtime configuration menu.

    Manage the interactive CLI flow for collecting and validating runtime
    arguments and configuration selections.

    Attributes:
        _args (dict): Current in-progress arguments.
        _config_args (dict): Loaded configuration arguments from the selected config.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        defaults: dict = None,
        reload  : bool = True,
        verbose : bool = False,
    ):
        """Initialize a new instance.

        Args:
            defaults (dict): Default overrides for arguments. Defaults to None.
            reload (bool): Reload configuration files. Defaults to True.
            verbose (bool): If True, enable verbose logging. Defaults to False.
        """
        self.verbose = verbose
        self.reload  = reload
        self._index  = 0
        self._args   = DEFAULT_ARGS
        self._args.update(defaults or {})
        self._config_args = {}
        # Setup project resolver
        self._project_resolver = ProjectResolver(
            root    = self._args.root,
            verbose = self.verbose
        )

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the total number of interactive steps."""
        return 27

    # --- Properties ---
    @property
    def args(self) -> dict:
        """Return current in-progress arguments."""
        return self._args

    @property
    def config_args(self) -> dict:
        """Return loaded configuration arguments from the selected config."""
        return self._config_args

    # --- Callable & Context Manager ---
    def prompt(self) -> dict | box.Box:
        """Run the interactive menu until completion.

        Returns:
            Final arguments mapping.
        """
        while True:
            self._display_prompt()
            if self._index == self.__len__():
                return self.args
            self._next()

    def _display_prompt(self):
        """Display and handle the current prompt step based on index."""
        if self._index == 0:
            # clear_terminal()
            console.rule(f"[bold red]Input Prompts")
        else:
            console.rule()

        if self._index == 0:  # Task
            self._args.task = Prompt(
                text    = CLI_OPTIONS["task"]["prompt_text"],
                default = self._args.task,
                choices = self._project_resolver.list_tasks(reload=self.reload),
            ).prompt()
        if self._index == 1:  # Mode
            self._args.mode = Prompt(
                text    = CLI_OPTIONS["mode"]["prompt_text"],
                default = self._args.mode,
                choices = CLI_OPTIONS["mode"]["choices"],
            ).prompt()
        if self._index == 2:  # Arch
            self._args.arch = Prompt(
                text    = CLI_OPTIONS["arch"]["prompt_text"],
                default = self._args.arch,
                choices = self._project_resolver.list_archs(
                    task   = self._args.task,
                    mode   = self._args.model,
                    reload = self.reload,
                ),
            ).prompt()
        if self._index == 3:  # Model
            self._args.model = Prompt(
                text    = CLI_OPTIONS["model"]["prompt_text"],
                default = self._args.model,
                choices = self._project_resolver.list_models(
                    task   = self._args.task,
                    mode   = self._args.mode,
                    arch   = self._args.arch,
                    reload = self.reload,
                )
            ).prompt()
        if self._index == 4:  # Config
            self._args.config = Prompt(
                text    = CLI_OPTIONS["config"]["prompt_text"],
                default = self._args.config,
                choices = self._project_resolver.list_config_files(
                    model_root    = parse_model_dir(self._args.arch, self._args.model),
                    model         = self._args.model,
                    absolute_path = True,
                    reload        = self.reload,
                )
            ).prompt()
            self._config_args = load_config(self._args.config, self.verbose)
        if self._index == 5:  # Weights
            self._args.weights = WeightsPrompt(
                text    = CLI_OPTIONS["weights"]["prompt_text"],
                default = parse_weights_file(
                    root    = self._args.root,
                    weights = self._args.weights or self._config_args.get("weights"),
                ),
                choices = self._project_resolver.list_weights_files(
                    model  = self._args.model,
                    reload = self.reload,
                ),
            ).prompt()
        if self._index == 6:  # Data
            if self._args.mode not in ["predict"]:
                self._next()
            else:
                self._args.data = DataPrompt(
                    text    = CLI_OPTIONS["data"]["prompt_text"],
                    default = self._args.data,
                    choices = self._project_resolver.list_datasets(
                        task   = self._args.task,
                        mode   = self._args.mode,
                        reload = self.reload,
                    ),
                ).prompt()
        if self._index == 7:  # Fullname
            _config  = self._args.config
            _model   = self._args.model
            _default = self._args.fullname or self._config_args.get("fullname")
            _default = _default or (Path(_config).stem if _config not in [None, "None", ""] else _model)
            self._args.fullname = Prompt(
                text    = CLI_OPTIONS["fullname"]["prompt_text"],
                default = _default,
            ).prompt()
        if self._index == 8:  # Device
            self._args.device = Prompt(
                text    = CLI_OPTIONS["device"]["prompt_text"],
                default = self._args.device or "cuda:0",
                choices = CLI_OPTIONS["device"]["choices"],
            ).prompt()
        if self._index == 9:  # Seed
            self._args.seed = NumberPrompt(
                text    = CLI_OPTIONS["seed"]["prompt_text"],
                default = self._args.seed or self._config_args.get("seed"),
            ).prompt()
        if self._index == 10:  # Image Size
            if self._args.mode not in ["predict", "speed"]:
                self._next()
            else:
                self._args.imgsz = NumberPrompt(
                    text    = CLI_OPTIONS["imgsz"]["prompt_text"],
                    default = self._args.imgsz or self._config_args.get("imgsz"),
                ).prompt()
        if self._index == 11:  # Epochs
            if self._args.mode not in ["train"]:
                self._next()
            else:
                self._args.epochs = NumberPrompt(
                    text    = CLI_OPTIONS["epochs"]["prompt_text"],
                    default = self._args.epochs or self._config_args.get("epochs"),
                ).prompt()
        if self._index == 12:  # Batch Size
            if self._args.mode not in ["train"]:
                self._next()
            else:
                self._args.batch_size = NumberPrompt(
                    text    = CLI_OPTIONS["batch_size"]["prompt_text"],
                    default = self._args.batch_size or self._config_args.get("batch_size"),
                ).prompt()
        if self._index == 13:  # torchrun
            if self._args.mode not in ["train"]:
                self._next()
            else:
                self._args.torchrun = Confirm(
                    text    = CLI_OPTIONS["torchrun"]["prompt_text"],
                    default = self._args.torchrun or self._config_args.get("torchrun", False),
                ).prompt()
        if self._index == 14:  # Master Port
            if self._args.mode not in ["train"] or not self._args.torchrun:
                self._next()
            else:
                self._args.master_port = NumberPrompt(
                    text    = CLI_OPTIONS["master_port"]["prompt_text"],
                    default = self._args.master_port or self._config_args.get("master_port"),
                ).prompt()
        if self._index == 15:  # Master Address
            if self._args.mode not in ["train"] or not self._args.torchrun:
                self._next()
            else:
                self._args.master_addr = Prompt(
                    text    = CLI_OPTIONS["master_addr"]["prompt_text"],
                    default = self._args.master_addr or self._config_args.get("master_addr"),
                ).prompt()
        if self._index == 16:  # Resize
            if self._args.mode not in ["predict", "speed"]:
                self._next()
            else:
                self._args.resize = Confirm(
                    text    = CLI_OPTIONS["resize"]["prompt_text"],
                    default = self._args.resize or self._config_args.get("resize", False),
                ).prompt()
        if self._index == 17:  # Benchmark
            self._args.benchmark = Confirm(
                text    = CLI_OPTIONS["benchmark"]["prompt_text"],
                default = self._args.benchmark or self._config_args.get("benchmark", False),
            ).prompt()
        if self._index == 18:  # Save Result
            if self._args.mode in ["speed"]:
                self._args.save_result = Confirm(
                    text    = CLI_OPTIONS["save_result"]["prompt_text"],
                    default = self._args.save_result,
                ).prompt()
            else:
                self._args.save_result = Confirm(
                    text    = CLI_OPTIONS["save_result"]["prompt_text"],
                    default = self._args.save_result or self._config_args.get("save_result", False),
                ).prompt()
        if self._index == 19:  # Save Image
            if self._args.mode in ["speed"]:
                self._args.save_image = Confirm(
                    text    = CLI_OPTIONS["save_image"]["prompt_text"],
                    default = self._args.save_image,
                ).prompt()
            else:
                self._args.save_image = Confirm(
                    text    = CLI_OPTIONS["save_image"]["prompt_text"],
                    default = self._args.save_image or self._config_args.get("save_image", False),
                ).prompt()
        if self._index == 20:  # Save Debug
            if self._args.mode in ["speed"]:
                self._args.save_debug = Confirm(
                    text    = CLI_OPTIONS["save_debug"]["prompt_text"],
                    default = self._args.save_debug,
                ).prompt()
            else:
                self._args.save_debug = Confirm(
                    text    = CLI_OPTIONS["save_debug"]["prompt_text"],
                    default = self._args.save_debug or self._config_args.get("save_debug", False),
                ).prompt()
        if self._index == 21:  # Use Fullname
            self._args.use_fullname = Confirm(
                text    = CLI_OPTIONS["use_fullname"]["prompt_text"],
                default = self._args.use_fullname or self._config_args.get("use_fullname", False),
            ).prompt()
        if self._index == 22:  # Keep Subdirs
            self._args.keep_subdirs = Confirm(
                text    = CLI_OPTIONS["keep_subdirs"]["prompt_text"],
                default = self._args.keep_subdirs or self._config_args.get("keep_subdirs", False),
            ).prompt()
        if self._index == 23:  # Save Nearby
            if self._args.mode not in ["predict"]:
                self._next()
            else:
                self._args.save_nearby = Confirm(
                    text    = CLI_OPTIONS["save_nearby"]["prompt_text"],
                    default = self._args.save_nearby or self._config_args.get("save_nearby", False),
                ).prompt()
        if self._index == 24:  # Exist OK?
            self._args.exist_ok = Confirm(
                text    = CLI_OPTIONS["exist_ok"]["prompt_text"],
                default = self._args.exist_ok or self._config_args.get("exist_ok", False),
            ).prompt()
        if self._index == 25:  # Use Verbose
            self._args.verbose = Confirm(
                text    = CLI_OPTIONS["verbose"]["prompt_text"],
                default = self._args.verbose or self._config_args.get("verbose", False),
            ).prompt()
        if self._index == 26:  # Finish
            rprint_dict(self._args, title="Input Arguments")
            finish = Confirm(text="Finish/Re-input", default=True).prompt()
            if finish:
                self._index = self.__len__()

    def _next(self):
        """Advance the prompt index by one."""
        self._index = (self._index + 1) % self.__len__()

    def _prev(self):
        """Move the prompt index back by one."""
        self._index = (self._index - 1) % self.__len__()

# endregion
