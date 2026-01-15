#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runtime utilities.

This module provides command-line interface (CLI) parsing, interactive prompts,
and configuration management for machine learning tasks. It includes classes
and functions to handle user input, load configurations, and manage runtime
arguments.
"""

from __future__ import annotations

__all__ = [
    "CLI_OPTIONS",
    "DEFAULT_ARGS",
    "ICLI",
    "ProjectResolver",
    "load_config",
    "parse_cli_args",
    "parse_default_args",
    "parse_model_fullname",
    "parse_predict_args",
    "parse_train_args",
    "print_run_summary",
    "resolve_config_file",
    "resolve_data_dir",
    "resolve_model_dir",
    "resolve_output_dir",
    "resolve_save_dir",
    "resolve_weights",
    "resolve_weights_dir",
    "resolve_weights_file",
]

import argparse
import importlib.util
import socket
from typing import Any, Callable, Optional, TypeVar

import box
import yaml
from rich import prompt

from mon.core.console import console, log, log_error, pprint_dict, rprint_dict
from mon.core.constants import MONO_ROOT_DIR, ROOT_DIR, ZOO_DIR
from mon.core.device import list_devices, parse_device
from mon.core.dtypes import image as I, Weights
from mon.core.enum import RunMode, Task, TRTPrecision
from mon.core.factory import DATASETS, MODELS, WEIGHTS
from mon.core.pathlib import Path
from mon.core.rich import SelectionOrInputPrompt
from mon.core.utils import (
    depascalize,
    is_int,
    merge_dicts,
    to_int,
    to_list,
    to_str,
)


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# --- Utilities ---

T = TypeVar("T")


def _is_null(value: Any) -> bool:
    """Check if a value should be treated as a Python None.

    Args:
        value: Value to check.
    """
    # Added .strip() check for string types to catch "  "
    if isinstance(value, str):
        value = value.strip()
    return value in [None, "None", "none", "NULL", ""]


def _safe_convert(value: Any, constructor: Callable[[Any], T]) -> T | None:
    """Convert a value using a constructor with null checking.

    Args:
        value: Value to convert.
        constructor: Constructor function to use for conversion.
    """
    if _is_null(value):
        return None
    try:
        # Special case: int("1.0") fails in Python, so we float() first
        if constructor is int:
            return int(float(value))
        return constructor(value)
    except (ValueError, TypeError):
        return None


def _str_or_none(value: Any) -> str | None:
    """Convert a value to a string or None.

    Args:
        value: Value to convert.
    """
    return _safe_convert(value, str)


def _int_or_none(value: Any) -> int | None:
    """Convert a value to an integer or None.

    Args:
        value: Value to convert.
    """
    return _safe_convert(value, int)


def _float_or_none(value: Any) -> float | None:
    """Convert a value to a float or None.

    Args:
        value: Value to convert.
    """
    return _safe_convert(value, float)


# --- Option Registry ---

CLI_OPTIONS = {
    "p"            : {
        "action"     : "store_true",
        "help"       : "Run with interactive prompt.",
        "prompt_only": False,
        "prompt_text": "",
    },
    # Basic
    "root"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Project root.",
        "prompt_only": False,
        "prompt_text": "Project Root",
    },
    "task"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : Task.values(),
        "help"       : f"Task to run: {Task.values()}.",
        "prompt_only": False,
        "prompt_text": "Task",
    },
    "mode"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : RunMode.values(),
        "help"       : f"Run mode: {RunMode.values()}.",
        "prompt_only": False,
        "i_cli_type" : str,
        "prompt_text": "Run Mode",
    },
    "arch"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Model architecture.",
        "prompt_only": False,
        "prompt_text": "Architecture",
    },
    "model"        : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Model name.",
        "prompt_only": False,
        "prompt_text": "Model",
    },
    "config"       : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Config file.",
        "prompt_only": False,
        "prompt_text": "Config",
    },
    "data"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Dataset name or directory.",
        "prompt_only": False,
        "prompt_text": "Predict(s)",
    },
    "fullname"     : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Full name of the current run.",
        "prompt_only": False,
        "prompt_text": "Fullname",
    },
    "save_dir"     : {
        "type"       : _str_or_none,
        "default"    : None,
        "help"       : "Directory to save the outputs.",
        "prompt_only": False,
        "prompt_text": "Save Directory",
    },
    "weights"      : {
        "action"     : "append",
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Path(s) to the pretrained weights.",
        "prompt_only": False,
        "prompt_text": "Weights",
    },
    "device"       : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : list_devices(),
        "help"       : f"Running device: {list_devices()}.",
        "prompt_only": False,
        "prompt_text": "Device",
    },
    "seed"         : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Seed.",
        "prompt_only": False,
        "prompt_text": "Seed         ",
    },
    "imgsz"        : {
        "action"     : "append",
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Image size.",
        "prompt_only": False,
        "prompt_text": "Image Size   ",
    },
    # Train
    "epochs"       : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Training epochs.",
        "prompt_only": False,
        "prompt_text": "Epochs       ",
    },
    "batch_size"   : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Batch size.",
        "prompt_only": False,
        "prompt_text": "Batch Size   ",
    },
    "torchrun"     : {
        "default"    : False,
        "action"     : "store_true",
        "help"       : "Using torch distributed training.",
        "prompt_only": False,
        "prompt_text": "Use torchrun?",
    },
    "master_port"  : {
        "default"    : 7777,
        "type"       : _int_or_none,
        "help"       : "Port for distributed communication.",
        "prompt_only": False,
        "prompt_text": "Master Port",
    },
    "master_addr"  : {
        "default"    : "localhost",
        "type"       : _str_or_none,
        "help"       : "Master node address.",
        "prompt_only": False,
        "prompt_text": "Master Address",
    },
    "local_rank"   : {
        "type"       : _int_or_none,
        "help"       : "Local rank for distributed training.",
        "prompt_only": False,
        "prompt_text": "Local Rank   ",
    },
    # Predict
    "resize"       : {
        "action"     : "store_true",
        "help"       : "Resize the input image.",
        "prompt_only": False,
        "prompt_text": "Resize?      ",
    },
    "benchmark"    : {
        "action"     : "store_true",
        "help"       : "Enable benchmark mode.",
        "prompt_only": False,
        "prompt_text": "Benchmark?   ",
    },
    # Save & Visualize
    "save_result"  : {
        "action"     : "store_true",
        "help"       : "Save results.",
        "prompt_only": False,
        "prompt_text": "Save Result? ",
    },
    "save_image"   : {
        "action"     : "store_true",
        "help"       : "Save output images.",
        "prompt_only": False,
        "prompt_text": "Save Image?  ",
    },
    "save_debug"   : {
        "action"     : "store_true",
        "help"       : "Save debug information.",
        "prompt_only": False,
        "prompt_text": "Save Debug?  ",
    },
    "use_fullname" : {
        "action"     : "store_true",
        "help"       : "Use the ``fullname`` for the ``save_dir``.",
        "prompt_only": False,
        "prompt_text": "Use Fullname?",
    },
    "keep_subdirs" : {
        "action"     : "store_true",
        "help"       : "Keep subdirectories in the ``save_dir``.",
        "prompt_only": False,
        "prompt_text": "Keep Subdirs?",
    },
    "save_nearby"  : {
        "action"     : "store_true",
        "help"       : "Save outputs nearby the source.",
        "prompt_only": False,
        "prompt_text": "Save Nearby? ",
    },
    "exist_ok"     : {
        "action"     : "store_true",
        "help"       : "Keep existing directories.",
        "prompt_only": False,
        "prompt_text": "Exist OK?    ",
    },
    "verbose"      : {
        "action"     : "store_true",
        "help"       : "Verbose mode.",
        "prompt_only": False,
        "prompt_text": "Verbosity?   ",
    },
    # Export
    "trt_precision": {
        "default"    : "fp32",
        "type"       : _str_or_none,
        "choices"    : TRTPrecision.values(),
        "help"       : f"TRT precision: {TRTPrecision.values()}.",
        "prompt_only": False,
        "prompt_text": "TRT Precision",
    },
}
CLI_OPTIONS = box.Box(CLI_OPTIONS)


# --- State Initialization ---

DEFAULT_ARGS = {
    k: False if v.get("action") in ["store_true"] else v.get("default", None)
    for k, v in CLI_OPTIONS.items()
}
DEFAULT_ARGS = box.Box(DEFAULT_ARGS)

# endregion


# ==============================================================================
# region CLI
# ==============================================================================

# --- Prompts ---

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


# --- CLIs ---

class ICLI:
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
                    model_root    = resolve_model_dir(self._args.arch, self._args.model),
                    model         = self._args.model,
                    absolute_path = True,
                    reload        = self.reload,
                )
            ).prompt()
            self._config_args = load_config(self._args.config, self.verbose)
        if self._index == 5:  # Weights
            self._args.weights = WeightsPrompt(
                text    = CLI_OPTIONS["weights"]["prompt_text"],
                default = resolve_weights_file(
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


# --- Parsing ---

def parse_default_args(name: str = "main") -> box.Box:
    """Build and parse default CLI arguments.

    Construct an argparse.ArgumentParser from ``CLI_OPTIONS`` and return
    parsed arguments.

    Args:
        name: Program description used in the ArgumentParser. Defaults to "main".

    Returns:
        Parsed arguments as a box.Box.
    """
    parser = argparse.ArgumentParser(description=name)

    for opt_name, opt_params in CLI_OPTIONS.items():
        if opt_params.get("prompt_only", False):
            continue

        action = opt_params.get("action", "store")
        kwargs = {
            "action"  : action,
            "help"    : opt_params.get("help", ""),
            "required": opt_params.get("required", False),
        }

        # Boolean actions (store_true/store_false) do not take 'type' or 'choices'
        if action in ["store_true", "store_false"]:
            # Default is usually False for store_true, True for store_false
            kwargs["default"] = opt_params.get("default", action == "store_false")
        else:
            if "type" in opt_params:
                kwargs["type"] = opt_params["type"]
            if "choices" in opt_params:
                kwargs["choices"] = opt_params["choices"]
            kwargs["default"] = opt_params.get("default", None)

        flag = f"--{opt_name.replace('_', '-')}"
        parser.add_argument(flag, **kwargs)

    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")
    return box.Box(vars(parser.parse_args()))


def parse_cli_args(
    cli : box.Box    | None = None,
    root: Path | str | None = None,
    name: str               = "main"
) -> box.Box:
    """Parse CLI arguments and optionally run the interactive prompt.

    Launch RunCLI to gather values if the ``p`` flag is present.

    Args:
        cli: Pre-parsed CLI arguments. Defaults to None.
        root: Project root to attach to parsed arguments. Defaults to None.
        name: Program description for the parser. Defaults to "main".

    Returns:
        Normalized CLI arguments.
    """
    # Initialize CLI if not provided
    cli = cli or parse_default_args(name)

    # Path Normalization
    # Prioritize root passed to function, then root in cli, then current working dir
    raw_root = root or cli.get("root") or Path.cwd()
    cli.root = Path(raw_root).normalize()

    # Interactive Switch
    # Assuming 'p' is the flag for --prompt
    if cli.get("p", False):
        # RunCLI should return a updated box.Box
        cli   = ICLI(cli).prompt()
        cli.p = False  # Prevent re-triggering

    return cli


def parse_train_args(
    cli       : box.Box    | None = None,
    root      : Path | str | None = None,
    model_root: Path | str | None = None,
    verbose   : bool              = False
) -> box.Box:
    """Parse and prepare training arguments.

    Merge ``cli`` and configuration values, resolve paths and devices, and
    prepare the save directory.

    Args:
        cli: CLI arguments. Defaults to None.
        root: Project root path. Defaults to None.
        model_root: Model root path for configuration resolution. Defaults to None.
        verbose: Verbosity mode. Defaults to False.

    Returns:
        Finalized training arguments.
    """
    # Resolve CLI and Config Path
    cli         = parse_cli_args(cli, root=root)
    config_path = resolve_config_file(cli.config, cli.root, model_root=model_root)

    # Load and Merge
    args = load_config(config_path, verbose=verbose)
    args = merge_dicts(args, cli)  # Prioritize cli -> args

    # Name and Directory Resolution
    args.fullname = args.fullname or args.model or "unnamed_run"

    if not args.save_dir:
        base_run_dir  = args.root / "run" / "train"
        # Determine subdir based on user preference
        subdir        = args.fullname if args.use_fullname else args.data
        args.save_dir = resolve_save_dir(base_run_dir, args.arch, args.model, subdir)
    else:
        args.save_dir = Path(args.save_dir)

    # Resource Resolution
    args.hostname = socket.gethostname().lower()
    args.device   = parse_device(args.device)
    # Resolve all potential weight paths
    for key in ["weights", "resume", "tuning"]:
        if key in args:
            args[key] = resolve_weights(
                root        = args.root,
                weights     = args[key],
                num_classes = args.num_classes,
            )

    # Save Directory Preparation (Atomic & Safe)
    if args.save_dir.exists() and not args.exist_ok:
        args.save_dir.rmdir(recursive=True)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Artifact Logging
    if config_path and config_path.exists():
        # Copying the config to the run dir ensures reproducibility
        config_path.copy_to(dst=args.save_dir / config_path.name)
        args.cli = config_path

    if verbose:
        console.log(f"[green]Run directory:[/green] {args.save_dir}")

    return args


def parse_predict_args(
    cli       : box.Box    | None = None,
    root      : Path | str | None = None,
    model_root: Path | str | None = None,
    verbose   : bool              = False
) -> box.Box:
    """Parse and prepare prediction arguments.

    Merge ``cli`` and configuration values, resolve devices and weights, and
    adjust image size.

    Args:
        cli: CLI arguments. Defaults to None.
        root: Project root path. Defaults to None.
        model_root: Model root path for configuration resolution. Defaults to None.
        verbose: Verbosity mode. Defaults to False.

    Returns:
        Finalized prediction arguments.
    """
    # Resolve CLI and Config Path
    cli         = parse_cli_args(cli, root=root)
    config_path = resolve_config_file(cli.config, cli.root, model_root=model_root)

    # Load and Merge
    args = load_config(cli.config, verbose=verbose)
    args = merge_dicts(args, cli)  # Prioritize cli -> args

    # Name and Directory Resolution
    args.fullname = args.fullname or args.model or "unnamed_prediction"

    if not args.save_dir:
        base_run_dir  = args.root / "run" / "predict"
        # Determine subdir grouping
        subdir        = args.fullname if (args.use_fullname or args.save_nearby) else args.data
        args.save_dir = resolve_save_dir(base_run_dir, args.arch, args.model, subdir)
    else:
        args.save_dir = Path(args.save_dir)

    # Resource Resolution
    args.hostname = socket.gethostname().lower()
    args.device   = parse_device(args.device)
    # Resolve all potential weight paths
    for key in ["weights", "resume", "tuning"]:
        if key in args:
            args[key] = resolve_weights(
                root        = args.root,
                weights     = args[key],
                num_classes = args.num_classes,
            )
    # Ensure imgsz is a list/tuple of [H, W] or a single int normalized to [H, W]
    args.imgsz = I.imgsz(args.imgsz)

    # Save Logic (Conditional for Inference)
    # Only create directories if we actually intend to save something and aren't saving 'nearby' the source
    should_save = any([args.save_result, args.save_image, args.save_debug])

    if not args.save_nearby and should_save:
        if args.save_dir.exists() and not args.get("exist_ok", False):
            args.save_dir.rmdir(recursive=True)

        args.save_dir.mkdir(parents=True, exist_ok=True)

        # Artifact Logging
        if config_path and config_path.exists():
            # Copying the config to the run dir for reproducibility of prediction settings
            config_path.copy_to(dst=args.save_dir / config_path.name)
            cli.config = config_path

    if verbose:
        console.log(f"[green]Run directory:[/green] {args.save_dir}")

    return args

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================

class ProjectResolver:
    """Resolves project configurations and settings."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root   : Path | str | None = None,
        verbose: bool       | None = False,
        *args, **kwargs
    ):
        self.verbose = verbose
        self.root    = root

        # Stored caches
        self._defaults      = {}
        self._models        = []
        self._archs         = []
        self._tasks         = []
        self._config_files  = []
        self._weights_files = []
        self._datasets      = []
        # Load cached values (reducing I/O overhead)
        self.load_defaults()
        self.list_tasks()

    # --- Properties ---
    @property
    def root(self) -> Path | None:
        """Return the project root directory."""
        return self._root

    @root.setter
    def root(self, value: Path | str | None):
        """Set the project root directory."""
        if isinstance(value, (Path, str)):
            value = Path(value).normalize()
            if not value.exists():
                value = None

        self._root = value

    @property
    def defaults(self) -> dict:
        """Return the project's default configurations."""
        return self._defaults if self._defaults else self.load_defaults()

    @property
    def tasks(self) -> list[str]:
        """Return a list of available tasks."""
        return self._tasks if self._tasks else self.list_tasks()

    @property
    def models(self) -> list[str]:
        """Return a list of available models."""
        return self._models if self._models else self.list_models()

    @property
    def archs(self) -> list[str]:
        """Return a list of available architectures."""
        return self._archs if self._archs else self.list_archs()

    @property
    def config_files(self) -> list[Path]:
        """Return a list of available configuration files."""
        return self._config_files if self._config_files else self.list_config_files()

    @property
    def weights_files(self) -> list[Path]:
        """Return a list of available weights."""
        return self._weights_files if self._weights_files else self.list_weights_files()

    @property
    def datasets(self) -> list[str]:
        """Return a list of available datasets."""
        return self._datasets if self._datasets else self.list_datasets()

    # --- Discovery ---
    def list_tasks(self, reload: bool = False) -> list[str]:
        """List available tasks supported by the project.

        Args:
            reload: If True, force a reload of the model registry. Defaults to False.

        Returns:
            Sorted list of task names.
        """
        # If reload is True, clear the cache
        if reload:
            self._tasks = []

        # Return cached tasks if available
        if self._tasks:
            return self._tasks

        # Start with global defaults
        # Assuming Task.names() returns a list of Enum objects or strings
        tasks = Task.names()

        # Consult project-specific restrictions
        project_tasks = self._defaults.get("TASKS")
        # Ensure we have a non-empty list
        if project_tasks:
            tasks = project_tasks

        # Normalize to string values and remove duplicates
        # This handles both Enum objects (t.value) and raw strings
        output = set()
        for t in tasks:
            val = t.value if hasattr(t, "value") else str(t)
            output.add(val.lower().strip())

        # Sort alphabetically
        output = sorted(list(output))

        # Cache the results
        if self._tasks is None or self._tasks != output:
            self._tasks = output

        return self._tasks

    def list_models(
        self,
        task  : str | None = None,
        mode  : str | None = None,
        arch  : str | None = None,
        reload: bool       = False,
    ) -> list[str]:
        """List available models for a given task, mode, and architecture.

        Args:
            task: Task name to filter models. Defaults to None.
            mode: Run mode to filter models. Defaults to None.
            arch: Architecture name to filter models. Defaults to None.
            reload: If True, force a reload of the model registry. Defaults to False.

        Returns:
            Sorted list of model names.
        """
        # If reload is True, clear the cache
        if reload:
            self._models = []

        # Return cached models if available
        if self._models:
            return self._models

        models = MODELS.filter(task=task, mode=mode, arch=arch)

        # Apply Project Scoping (Restrict models to specific project-approved versions)
        project_allowed = self._defaults.get("MODELS")
        if project_allowed:
            # Normalize for case-insensitive matching
            project_set = {depascalize(m) for m in project_allowed}
            models      = [m for m in models if depascalize(m) in project_set]

        # Sort alphabetically
        models = sorted(models)

        # Cache the results
        if self._models is None or self._models != models:
            self._models = models

        return self._models

    def list_archs(
        self,
        task  : str | None = None,
        mode  : str | None = None,
        reload: bool       = False,
    ) -> list[str]:
        """List available architectures for a given task and mode.

        Args:
            task: Task name to filter architectures. Defaults to None.
            mode: Run mode to filter architectures. Defaults to None.
            reload: If True, force a reload of the model registry. Defaults to False.

        Returns:
            Sorted list of architecture names.
        """
        # If reload is True, clear the cache
        if reload:
            self._archs = []

        # Return cached architectures if available
        if self._archs:
            return self._archs

        # Get the base model list
        models = self.list_models(task=task, mode=mode)

        # Resolve Architecture names from registry
        flattened_registry = MODELS.flatten_dict
        archs = set()  # Use set to automatically handle duplicates

        for m in models:
            # Check if model exists in registry and has an 'arch' attribute
            entry = flattened_registry.get(m)
            if entry and hasattr(entry, "arch"):
                a = str(entry.arch).strip()
                if a and a.lower() != "none":
                    archs.add(a)

        # Sort alphabetically
        archs = sorted(list(archs))

        # Cache the results
        if self._archs is None or self._archs != archs:
            self._archs = archs

        return self._archs

    def list_config_files(
        self,
        model_root   : Path | None = None,
        model        : str  | None = None,
        absolute_path: bool        = False,
        reload       : bool        = False,
    ) -> list[Path]:
        """List available running configuration files for the project.

        List configuration files found under project and model config
        directories and return either filenames or absolute paths.

        Args:
            model_root: Optional model-specific root to include. Defaults to None.
            model: Optional model name to filter results. Defaults to None.
            absolute_path: If True, return absolute Paths; otherwise return
                names. Defaults to False.
            reload: If True, force a reload of the model registry. Defaults to False.

        Returns:
            Sorted list of configuration file paths or filenames.
        """
        # If reload is True, clear the cache
        if reload:
            self._config_files = []

        # Return cached results if available
        if self._config_files:
            return self._config_files

        # Gather all potential files
        root         = self._root
        config_files = []
        if self.is_valid(root):
            config_files.extend(self.collect_config_files(root))
        if self.is_valid(model_root):
            config_files.extend(self.collect_config_files(model_root))

        # Filter by file type
        # keeps .yaml/.json (via is_config_file) and non-init .py files
        config_files = [
            cf for cf in config_files
            if cf.is_config_file() or (cf.suffix == ".py" and cf.name != "__init__.py")
        ]

        # Optional Model Filtering
        if self.is_valid(model):
            config_files = [cf for cf in config_files if model in cf.name]

        # Format Output
        results = [cf if absolute_path else cf.name for cf in config_files]

        # Remove duplicates and sort alphabetically
        results = sorted(list(set(results)))

        # Cache the results
        if self._config_files is None or self._config_files != results:
            self._config_files = results

        return self._config_files

    def list_weights_files(
        self,
        model : str | None = None,
        reload: bool       = False
    ):
        """List available weights for a given model."""
        # If reload is True, clear the cache
        if reload:
            self._weights_files = []

        # Return cached weights if available
        if self._weights_files:
            return self._weights_files

        weights = []

        # Collect from local training runs
        if self.is_valid(self._root):
            train_dir = self._root / "run" / "train"
            weights.extend(self.collect_weights(train_dir))

        # Collect from global Model Zoo
        if self.is_valid(ZOO_DIR):
            weights.extend(self.collect_weights(ZOO_DIR))

        # Filter by Model Name
        # We check parts to ensure the weight belongs to a folder/file named after the model
        if model:
            weights = [
                f.absolute() for f in weights
                if model.lower() in [p.lower() for p in f.parts]
            ]

        # Remove duplicates and sort alphabetically
        weights = sorted(list(set(weights)))

        # Cache the results
        if self._weights_files is None or self._weights_files != weights:
            self._weights_files = weights

        return self._weights_files

    def list_datasets(
        self,
        task  : str | None = None,
        mode  : str | None = None,
        reload: bool       = False
    ) -> list[str]:
        """List available datasets for a given task and mode.

        Args:
            task: Task name.
            mode: Run mode, e.g., "train" or "predict".
            reload: If True, force a reload of the dataset registry. Defaults to False.

        Returns:
            List of dataset names supporting the task and split.
        """
        # If reload is True, clear the cache
        if reload:
            self._datasets = []

        # Return cached datasets if available
        if self._datasets:
            return self._datasets

        datasets = DATASETS.filter(task=task, mode=mode)

        # Apply Project Scoping (Restrict models to specific project-approved versions)
        project_allowed = self._defaults.get("DATASETS")
        if project_allowed:
            # Normalize for case-insensitive matching
            project_set = {depascalize(m) for m in project_allowed}
            datasets    = [d for d in datasets if depascalize(d) in project_set]

        # Sort alphabetically
        datasets = sorted(datasets)

        # Cache the results
        if self._datasets is None or self._datasets != datasets:
            self._datasets = datasets

        return self._datasets

    # --- Input ---
    def load_defaults(self) -> dict:
        """Load default configurations from the project root directory.

        Read the project's config/default.py and return the defined defaults.

        Returns:
            Defaults mapping loaded from the project's default.py, or an empty
            mapping if none is present.
        """
        root = self._root

        # Validate Input
        if not root:
            return {}

        config_file = root / "config" / "default.py"

        # Check existence
        if not config_file.is_file():
            return {}

        # Dynamic Execution
        try:
            spec = importlib.util.spec_from_file_location("project_defaults", str(config_file))
            if spec is None or spec.loader is None:
                return {}

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extraction & Cleanup
            # Filtering for uppercase constants is usually safer to avoid
            # picking up imported modules (like 'import os')
            return {
                key: value
                for key, value in module.__dict__.items()
                if key.isupper() and not key.startswith('__')
            }
        except (ImportError, SyntaxError, AttributeError) as e:
            # Log the error but return empty dict to prevent total system crash
            return {}

    # --- Utils ---
    @staticmethod
    def is_valid(x) -> bool:
        return x is not None and str(x).lower() not in ["", "none"]

    @staticmethod
    def collect_config_files(root: Path) -> list[Path]:
        config_dir = (Path(root) / "config").normalize()
        if not config_dir.exists():
            return []

        found = []
        for file_path in config_dir.files(recursive=True):
            # Convert to parts to check directory names accurately
            parts = file_path.parts
            if "archive" in parts or "excluded" in parts:
                continue
            found.append(file_path)
        return found

    @staticmethod
    def collect_weights(root: Path) -> list[Path]:
        root = Path(root).normalize()
        if not root.exists():
            return []
        # Optimization: rglob with specific extensions if is_weights_file permits
        # Otherwise, stick to * but ensure it's a file
        return [f for f in root.rglob("*") if f.is_file(exist=True) and f.is_weights_file(exist=True)]

# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

def load_config(config: Any, verbose: bool = True) -> dict | box.Box:
    """Load configuration from a path, module, or mapping.

    Accept a mapping, a box.Box, or a filesystem path pointing to a Python or
    YAML configuration and return a loaded configuration mapping.

    Args:
        config: Mapping or path to a config file.
        verbose: Verbosity mode. Defaults to True.

    Returns:
        Loaded configuration as a box.Box. Returns an empty box.Box if nothing
        is found.
    """
    data = None

    # Handle direct objects
    if isinstance(config, box.Box):
        data = config
    elif isinstance(config, dict):
        data = box.Box(config)

    # Handle file paths
    elif isinstance(config, (Path, str)):
        config_path = Path(config)

        if config_path.suffix == ".py" and config_path.exists():
            spec   = importlib.util.spec_from_file_location(config_path.stem, str(config_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Filter: Only keep variables that are UPPERCASE and not internal
            # This prevents 'import torch' from cluttering your config
            data = {
                k: v for k, v in module.__dict__.items()
                if k.isupper() and not k.startswith("_")
            }

        elif config_path.suffix in [".yaml", ".yml"] and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                # SafeLoader is recommended unless you need Python object tags
                data = yaml.load(f, Loader=yaml.SafeLoader)

    # 3. Finalization and Logging
    if verbose:
        if data:
            log(f"Config successfully loaded from: {config}")
        else:
            log_error(f"No configuration found at {config}. Returning empty Box.")

    return box.Box(data or {})

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def parse_model_fullname(name: str, data: str, suffix: str | None = None) -> str:
    """Compose a model fullname from name, data, and optional suffix.

    Build a normalized fullname string by appending dataset and an
    optional suffix if not already present.

    Args:
        name: Base model name.
        data: Dataset or data identifier to append.
        suffix: Optional suffix to append. Defaults to None.

    Returns:
        Composed fullname string.
    """
    if not name or str(name).lower() == "none":
        # Using a default or raising is often better than just logging
        return "unnamed_model"

    # Start with the base architecture/model name
    fullname = str(name).strip()

    # Append dataset identifier
    if data and str(data).lower() != "none":
        data_tag = str(data).strip()
        if data_tag not in fullname:
            fullname = f"{fullname}_{data_tag}"

    # Append optional suffix (e.g., 'nano', 'pretrained', 'v2')
    if suffix and str(suffix).lower() != "none":
        # Normalize casing (e.g., 'Nano' -> 'nano')
        clean_suffix = depascalize(str(suffix).strip())

        # Avoid duplicate tags
        if clean_suffix not in fullname:
            fullname = f"{fullname}_{clean_suffix}"

    return fullname


def resolve_data_dir(root: Path | None, data_dir: Path | str = "") -> Path:
    """Resolve an absolute data directory path from candidates.

    Try a series of candidate locations and return the first existing
    directory. Raise an error if no candidate exists.

    Args:
        root: Project root.
        data_dir: Candidate data directory name or path. Defaults to "".

    Returns:
        First candidate directory path that exists.

    Raises:
        FileNotFoundError: If no candidate data directory is found.
    """
    # Standardize Inputs
    # Use global ROOT_DIR if no project root is provided
    root_path = Path(root).normalize(exist=True) if root else ROOT_DIR

    # Identify the target name/path
    target = Path(data_dir).normalize(exist=True) if data_dir else None

    # Build Ordered Candidates
    candidates = []

    if target:
        # If target is absolute, Path logic will prioritize it during joins
        candidates.extend([
            target,                            # Direct path
            root_path     / target,            # Relative to project root
            root_path     / "data" / target,   # Inside project data folder
            ROOT_DIR      / "data" / target,   # Inside "mon" data folder
            MONO_ROOT_DIR / "data" / target,   # Inside global data folder (monorepo)
        ])

    # Fallback search locations
    candidates.extend([
        root_path / "data",
        ROOT_DIR  / "data"
    ])

    # Validation Loop
    # Use unique paths only to avoid multiple disk IO checks on the same location
    seen = set()
    for d in candidates:
        abs_d = d.resolve() if d.is_absolute() else d.absolute()
        if abs_d not in seen:
            if abs_d.is_dir():
                return abs_d
            seen.add(abs_d)

    raise FileNotFoundError(
        f"Could not resolve data directory. Looked in: {[str(c) for c in candidates]}"
    )


def resolve_model_dir(arch: str, model: str) -> Optional[Path]:
    """Return the model directory for the given arch and model.

    Args:
        arch: Architecture name.
        model: Model name.

    Returns:
        Path to the model directory, or None if unspecified.
    """
    # Validation & Normalization
    if not arch or not model:
        return None

    # Registry Lookup with Safety
    try:
        # Access nested registry.
        # Using .get() allows for a more graceful failure than raw brackets.
        arch_entry = MODELS.get(arch)
        if arch_entry is None:
            return None

        model_entry = arch_entry.get(model)
        if model_entry is None:
            return None

        # Path Resolution
        model_dir = getattr(model_entry, "model_dir", None)

        if model_dir:
            return Path(model_dir)
    except Exception as e:
        # If logging is available, log the registry access failure
        return None

    return None


def resolve_save_dir(
    root : Path,
    arch : str | None = None,
    model: str | None = None,
    data : str | None = None,
) -> Path:
    """Build a save directory path from components.

    Combine root, architecture, model and optional data to construct a
    save directory path suitable for storing run outputs.

    Args:
        root: Base root path.
        arch: Optional architecture name. Defaults to None.
        model: Optional model name. Defaults to None.
        data: Optional data name or path. Defaults to None.

    Returns:
        Constructed save directory path.
    """
    # Start with the base root (e.g., 'project/runs/train')
    save_dir = Path(root).normalize()

    # Add Architecture level (e.g., 'yolov8')
    if arch and str(arch).lower() != "none":
        save_dir /= str(arch).lower().strip()

    # Add Model level (e.g., 'yolov8n')
    if model and str(model).lower() != "none":
        save_dir /= str(model).lower().strip()

        # Add Dataset level inside the model folder
        if data and str(data).lower() != "none":
            data_path = Path(data)
            # If it's a real path, take the filename (stem);
            # otherwise, use the string directly
            folder_name = data_path.stem if (data_path.suffix or data_path.exists()) else str(data)
            save_dir   /= folder_name.lower().strip()

    return save_dir


def resolve_output_dir(
    root        : Path,
    dirname     : Path | str,
    subdir_name : Path | str,
    src_path    : Path | str,
    keep_subdirs: bool = False,
    save_nearby : bool = False,
) -> Path:
    """Compute the output directory for a source path.

    Determine where to place outputs for a given source path, optionally
    preserving subdirectory structure or saving outputs near the source.

    Args:
        root: Base save root.
        dirname: Directory name used in save structure.
        subdir_name: Optional subdirectory under root to place outputs.
        src_path: Source file path used to preserve subdir structure.
        keep_subdirs: If True, preserve subdirectories from src_path.
            Defaults to False.
        save_nearby: If True, save outputs near the source path instead.
            Defaults to False.

    Returns:
        Resolved output directory path.
    """
    root        = Path(root).normalize()
    dirname     = Path(dirname)
    subdir_name = str(subdir_name) if subdir_name not in [None, "None", ""] else None
    src_path    = Path(src_path).normalize() if src_path else None

    # Logic for saving results next to the source file
    if save_nearby and src_path:
        # Create a folder like: path/to/image_results
        # Uses the stem of the root (e.g., 'predict') as a suffix
        suffix      = root.stem if root.stem != dirname.stem else root.parent.stem
        output_root = src_path.parent / f"{src_path.stem}_{suffix}"
        return output_root

    # Structure Preservation Logic
    if keep_subdirs and src_path:
        try:
            # Get path relative to the input root (dirname)
            # e.g., src: 'data/val/class1/img.jpg', dir: 'data' -> 'val/class1'
            rel_path    = src_path.parent.relative_to(dirname)
            target_path = root / rel_path
        except ValueError:
            # Fallback if src_path is not under dirname
            target_path = root / src_path.parent.name

        if subdir_name:
            return target_path / subdir_name
        return target_path

    # Default Centralized Logic
    # Nest by dirname if it's not already the root's name
    final_root = root
    if dirname.stem != root.stem:
        final_root = root / dirname.stem

    if subdir_name:
        return final_root / subdir_name
    return final_root


def resolve_weights_dir(root: Path, weights: Path | str) -> Path | None:
    """Resolve the weight directory from the given root and weights name or
    relative path.

    Args:
        root: Project root path.
        weights: Weights name or relative path.

    Returns:
        Absolute weights directory path or None if nothing was found.
    """
    root = Path(root).normalize(exist=True)
    # Ensure weights is always a Path object
    weights = Path(weights) if weights not in [None, "None", ""] else None

    # Check if the weight provided is already an absolute path
    if weights.is_absolute() and weights.is_dir():
        return weights

    # Check Local Project Root (Highest Priority)
    local_dir = root / weights
    if local_dir.is_dir():
        return local_dir

    # Check Global Zoo Directory
    global_dir = ZOO_DIR / weights
    if global_dir.is_dir():
        return global_dir

    # Return None if not found
    return None


def resolve_config_file(
    config      : Path,
    project_root: Path,
    model_root  : Path | None = None
) -> Path | None:
    """Resolve a config file path from given components.

    Search project and model config directories and return the first
    matching config file if found.

    Args:
        config: Candidate config name or path.
        project_root: Project root to search under.
        model_root: Optional model root to search under. Defaults to None.

    Returns:
        Resolved config path if found, otherwise None.
    """
    if not config or str(config).lower() == "none":
        return None

    config_path = Path(config).normalize()

    # Direct Path Check: If the user provided a valid absolute/relative path
    if config_path.exists() and config_path.is_file():
        return config_path

    # Define Search Hierarchy (Model-specific first, then Project-wide)
    search_roots = []
    if model_root:
        search_roots.append(Path(model_root) / "config")
    if project_root:
        search_roots.append(Path(project_root) / "config")

    # Search Loop
    for root in search_roots:
        if not root.is_dir():
            continue

        # Check the root of the config dir, then all subdirectories
        # Using rglob is more Pythonic for finding a specific filename recursively
        # We search for the exact name or the name with common config suffixes
        for candidate in root.rglob("*"):
            if candidate.is_file():
                # Check if it matches the name or the stem (if no suffix was provided)
                if candidate.name == config_path.name or candidate.stem == config_path.name:
                    # Assuming .is_config_file() validates the suffix internally
                    if hasattr(candidate, "is_config_file") and candidate.is_config_file():
                        return candidate
                    elif candidate.suffix in [".yaml", ".yml", ".py", ".json"]:
                        return candidate

    # Failure State
    log_error(f"Config not found: {config}. Searched in {search_roots}")
    return None


def resolve_weights_file(root: Path, weights: Path) -> Path | None:
    """Resolve the weight file from the given root and weights name or
    relative path.

    Args:
        root: Project root path.
        weights: Weights name or relative path.

    Returns:
        Absolute weight file path or None if nothing was found.
    """
    root = Path(root).normalize(exist=True)
    # Ensure weights is always a Path object
    weights = Path(weights) if weights not in [None, "None", ""] else None

    # Check if the weight provided is already an absolute path
    if weights.is_absolute() and weights.is_weights_file():
        return weights

    # Check Local Project Root (Highest Priority)
    # Search specifically for the file in the project's training runs
    local_file = root / weights
    if local_file.is_weights_file(exist=True):
        return local_file

    # Check Global Zoo Directory
    global_file = ZOO_DIR / weights
    if global_file.is_weights_file(exist=True):
        return weights

    # Return None if not found
    return None


def resolve_weights(
    root       : Path,
    weights    : Path,
    num_classes: int | None = None,
) -> Weights | None:
    """Resolve a ``Weights`` object from the given root and weights name or
    relative path.

    Args:
        root: Project root path.
        weights: Weights name or relative path.
        num_classes: Optional number of classes to set in the Weights object.
            Defaults to None.

    Returns:
        ``Weights`` object if found, otherwise None.
    """
    weights = resolve_weights_file(root=root, weights=weights)

    # If a valid weights file was found, wrap it in a Weights object
    if weights:
        # Check if the weights object is already registered in WEIGHTS
        if WEIGHTS.has(path=weights):
            return WEIGHTS.find_weights_objs(path=weights)
        # Otherwise, the weights object has not been registered yet.
        else:
            return Weights(path=weights, num_classes=num_classes)

    # Return None if not found
    return None


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region BASIC LOGGING
# ==============================================================================

def print_run_summary(args: dict | box.Box, full: bool = False):
    """Print a concise summary of run arguments.

    Print a compact run summary or the full configuration when requested.

    Args:
        args: Arguments mapping (box.Box or dict).
        full: If True, pretty-print the full args and config. Defaults to False.
    """
    # Handle Full Configuration Output
    if full:
        console.rule("[bold yellow]Full Configuration")
        # Ensure we have a standard dict for pretty printing
        printable_args = args.to_dict() if hasattr(args, "to_dict") else dict(args)
        pprint_dict(printable_args)
        return

    # Handle Concise Summary Output
    # We use .get() defaults to prevent crashes if certain keys are missing
    name = args.get("fullname", "Unnamed Run")
    console.rule(f"[bold red]{name}")
    summary_fields = {
        "Machine" : args.get("hostname", "local"),
        "Device"  : args.get("device", "cpu"),
        "Task"    : args.get("task"),
        "Mode"    : args.get("mode"),
        "Data"    : args.get("data"),
        "Weights" : args.get("weights"),
        "Save Dir": args.get("save_dir"),
        "Config"  : args.get("config"),
    }
    for label, value in summary_fields.items():
        if value:  # Only log fields that have a value
            # Formatting paths to be cleaner strings
            display_val = str(value) if not isinstance(value, list) else f"{len(value)} files"
            log(f"{label:<10}: {display_val}")

    console.rule() # Add a closing line for visual polish

# endregion
