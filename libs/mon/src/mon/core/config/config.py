#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config Data Structures & Management.

This module provides data structures and utilities for managing configuration
files.
"""

from __future__ import annotations

__all__ = [
    # "ARGUMENTS",
    "Config",
    "ConfigContext",
]

import argparse
import copy
import socket
from typing import Any, Callable, TypeVar

import torch
from box import Box

from mon.core.constants import DATASETS, MODELS, ZOO_ROOT
from mon.core.context import sys_ctx
from mon.core.data import create_weights
from mon.core.dtype import RunMode, Task
from mon.core.filesystem import (
    resolve_output_dir,
    resolve_save_dir,
    resolve_weights_file,
)
from mon.core.path import Path
from mon.core.typing import (
    DeviceLike,
    DictLike,
    PathLike,
    RunModeLike,
    TaskLike,
)
from mon.core.ui import (
    Confirm,
    console,
    log,
    log_error,
    OptionPrompt,
    PathPrompt,
    pprint_dict,
    Prompt,
)
from mon.core.utils import is_valid_str, merge_dicts

# ==============================================================================
# region CONSTANTS
# ==============================================================================

T = TypeVar("T")


def _is_null(value: Any) -> bool:
    """Check if a value should be treated as a Python None."""
    # Added .strip() check for string types to catch "  "
    if isinstance(value, str):
        value = value.strip()
    return value in [None, "None", "none", "NULL", ""]


def _safe_convert(value: Any, constructor: Callable[[Any], T]) -> T | None:
    """Convert a value using a constructor with null checking."""
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
    """Convert a value to a string or None."""
    return _safe_convert(value, str)


def _int_or_none(value: Any) -> int | None:
    """Convert a value to an integer or None."""
    return _safe_convert(value, int)


def _float_or_none(value: Any) -> float | None:
    """Convert a value to a float or None."""
    return _safe_convert(value, float)


ARGUMENTS = Box({
    # General
    "config": {
        "default": None,
        "type": _str_or_none,
        "help": "Config file.",
        "prompt_only": False,
        "prompt_text": "Config",
    },
    "exp_name": {
        "default": None,
        "type": _str_or_none,
        "help": "Experiment name.",
        "prompt_only": False,
        "prompt_text": "Experiment Name",
    },
    "root": {
        "default": None,
        "type": _str_or_none,
        "help": "Project root.",
        "prompt_only": False,
        "prompt_text": "Project Root",
    },
    "output_dir": {
        "type": _str_or_none,
        "default": None,
        "help": "Directory to save the outputs.",
        "prompt_only": False,
        "prompt_text": "Output Directory",
    },
    "task": {
        "default": None,
        "type": _str_or_none,
        "choices": [None] + Task.values(),
        "help": f"Task to run: {Task.values()}.",
        "prompt_only": False,
        "prompt_text": "Task",
    },
    "mode": {
        "default": None,
        "type": _str_or_none,
        "choices": [None] + RunMode.values(),
        "help": f"Run mode: {RunMode.values()}.",
        "prompt_only": False,
        "i_cli_type": str,
        "prompt_text": "Run Mode",
    },
    "arch": {
        "default": None,
        "type": _str_or_none,
        "help": "Model architecture.",
        "prompt_only": False,
        "prompt_text": "Architecture",
    },
    "model": {
        "default": None,
        "type": _str_or_none,
        "help": "Model name.",
        "prompt_only": False,
        "prompt_text": "Model",
    },
    "weights": {
        "action": "append",
        "default": None,
        "type": _str_or_none,
        "help": "Path(s) to the pretrained weights.",
        "prompt_only": False,
        "prompt_text": "Weights",
    },
    "data": {
        "default": None,
        "type": _str_or_none,
        "help": "Dataset name or directory.",
        "prompt_only": False,
        "prompt_text": "Predict(s)",
    },
    "device": {
        "default": None,
        "type": _str_or_none,
        "choices": [None] + sys_ctx.device_names,
        "help": f"Running device: {[None] + sys_ctx.device_names}.",
        "prompt_only": False,
        "prompt_text": "Device",
    },
    # Prediction
    "benchmark": {
        "action": "store_true",
        "help": "Enable benchmark mode.",
        "prompt_only": False,
        "prompt_text": "Benchmark?   ",
    },
    # Saving & Visualization
    "save": {
        "action": "store_true",
        "help": "Save results.",
        "prompt_only": False,
        "prompt_text": "Save Result? ",
    },
    "save_debug": {
        "action": "store_true",
        "help": "Save debug information.",
        "prompt_only": False,
        "prompt_text": "Save Debug?  ",
    },
    "keep_subdirs": {
        "action": "store_true",
        "help": "Keep subdirectories in the ``output_dir``.",
        "prompt_only": False,
        "prompt_text": "Keep Subdirs?",
    },
    "near_src": {
        "action": "store_true",
        "help": "Save the results near the source directory.",
        "prompt_only": False,
        "prompt_text": "Near Source? ",
    },
    "exist_ok": {
        "action": "store_true",
        "help": "Keep existing directories.",
        "prompt_only": False,
        "prompt_text": "Exist OK?    ",
    },
    "verbose": {
        "action": "store_true",
        "help": "Verbose mode.",
        "prompt_only": False,
        "prompt_text": "Verbosity?   ",
    },
})

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class Config:
    """A wrapper class for the configuration Box-like dictionary to provide
    a default schema and validation for the configuration attributes.
    """

    _DEFAULT_SCHEMA: dict = {
        # --- General ---
        "config_file": None,
        "hostname": "localhost",
        "exp_name": "",
        "root": None,
        "output_dir": None,
        "task": "",
        "mode": "",
        "device": None,
        "seed": 0,

        # --- Model ---
        "model": {
            "name": "",
            "arch": "",
            "weights": None,
            "finetune": None,
        },

        # --- Data ---
        "train_dataloader": {
            "dataset": {
                "name": "",
                "root": None,
                "dirname": "",
                "subdir": "",
                "split": "train",
                "transforms": {
                    "ops": [],
                    "p": 1.0,
                    "seed": None,
                },
                "modalities": [],
                "classes": None,
                "verbose": True,
            }
        },
        "val_dataloader": {
            "dataset": {
                "name": "",
                "root": None,
                "dirname": "",
                "subdir": "",
                "split": "val",
                "transforms": {
                    "ops": [],
                    "p": 1.0,
                }
            }
        },
        "data": [],

        # --- Training ---
        "epochs": 100,
        "optimizer": {
            "name": "adam",
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
        "lr_scheduler": {},
        "lr_warmup_scheduler": {},

        # --- Prediction ---
        "benchmark": False,

        # --- Saving & Visualization ---
        "save": True,
        "save_debug": False,
        "keep_subdirs": False,
        "near_src": False,
        "exist_ok": True,
        "verbose": True,
    }

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        config: DictLike | None = None,
        config_file: PathLike | None = None,
        root: PathLike | None = None,
        **kwargs
    ):
        """Initialize a new instance.

        Args:
            config (DictLike | None): A Box-like dictionary containing the
                initial configuration. Defaults to None.
            config_file (PathLike | None): Path to the configuration file.
                If given, it will be loaded and used to override the default
                configuration. Defaults to None.
            root (PathLike | None): Project root directory. Defaults to None.
            **kwargs: Additional keyword arguments for configuration updates.
        """
        # Allocate resources
        if not config:
            config = Box(self._DEFAULT_SCHEMA)
        elif isinstance(config, dict):
            config = Box(config)
        if not isinstance(config, Box):
            raise TypeError(
                f"Expected 'value' to be a Box-like dictionary, "
                f"but got {type(config).__name__}."
            )
        self._config: Box = config

        # Assign attributes
        self.root = root
        self.config_file = config_file
        self.cli_kwargs = kwargs

        if self.config_file:
            # If a config file is given, load it and update the default config
            self.update_from_yaml(self.config_file)
        if self.cli_kwargs:
            # If arguments are given from CLI, update the config
            self.update_from_cli(self.cli_kwargs)

    # --- Properties ---
    @property
    def config(self) -> Box:
        return self._config

    @property
    def config_file(self) -> Path | None:
        return self.config.config_file

    @config_file.setter
    def config_file(self, value: PathLike | None):
        # Validate inputs
        if not is_valid_str(value):
            return

        # Check if the given value is a valid path
        config_file = Path(value).normalize()
        if config_file.has_ext(".yaml", ".yml", exist=True):
            self._config.config_file = config_file
            return

        # Look for the configuration file in the project config directory
        if self.config_dir:
            config_file = self.config_dir / value
            if config_file.has_ext(".yaml", ".yml", exist=True):
                self._config.config_file = config_file
                return

    @property
    def exp_name(self) -> str:
        if self.config.exp_name:
            return self.config.exp_name
        elif self.config_file:
            return self.config_file.stem
        else:
            return self.model

    @exp_name.setter
    def exp_name(self, value: str):
        self._config.exp_name = value

    @property
    def root(self) -> Path:
        """Return the project root directory."""
        return self.config.root

    @root.setter
    def root(self, value: PathLike | None):
        """Set the project root directory."""
        # root = resolve_project_root(value)
        root = Path(value).normalize() if is_valid_str(value) else None
        if root and root.is_dir():
            # If the given value is a valid directory, set it as the root
           self._config.root = root

    @property
    def output_dir(self) -> Path | None:
        """Return the output directory."""
        return self._config.output_dir

    @output_dir.setter
    def output_dir(self, value: PathLike | None):
        """Set the output directory."""
        output_dir = Path(value).normalize() if is_valid_str(value) else None
        if output_dir and output_dir.is_dir():
            self._config.output_dir = output_dir

    @property
    def task(self) -> Task | None:
        return self._config.task

    @task.setter
    def task(self, value: TaskLike | None):
        if value in Task:
            self._config.task = Task(value)

    @property
    def mode(self) -> RunMode | None:
        return self._config.mode

    @mode.setter
    def mode(self, value: RunModeLike | None):
        if value in RunMode:
            self._config.mode = RunMode(value)

    @property
    def arch(self) -> str:
        return self._config.model.arch

    @arch.setter
    def arch(self, value: str | None):
        if is_valid_str(value):
            self._config.model.arch = value

    @property
    def model(self) -> str:
        return self._config.model.name

    @model.setter
    def model(self, value: str | None):
        if is_valid_str(value):
            self._config.model.name = value

    @property
    def weights(self) -> Path | None:
        return self.config.model.weights

    @weights.setter
    def weights(self, value: PathLike | None):
        if is_valid_str(value):
            self.config.model.weights = Path(value)

    @property
    def finetune(self) -> Path | None:
        return self._config.model.finetune

    @finetune.setter
    def finetune(self, value: PathLike | None):
        if is_valid_str(value):
            self._config.model.finetune = Path(value)

    @property
    def data(self) -> list[PathLike]:
        return self._config.data

    @data.setter
    def data(self, value: list[PathLike] | PathLike | None):
        # Normalize inputs
        data = []
        if isinstance(value, (Path, str)):
            data = [value]
        elif isinstance(value, list):
            data = value

        for i, d in enumerate(data):
            d_path = Path(d).normalize()
            if d_path.exists():
                # Path to a directory or file
                data[i] = d_path
            else:
                # Dataset name
                data[i] = d

        self._config.data = data

    @property
    def device(self) -> torch.device:
        return self._config.device

    @device.setter
    def device(self, value: DeviceLike):
        self._config.device = sys_ctx.get_torch_device(value)

    @property
    def seed(self) -> int:
        return self._config.seed

    @property
    def benchmark(self) -> bool:
        return self._config.benchmark

    @benchmark.setter
    def benchmark(self, value: bool):
        self._config.benchmark = value

    @property
    def save(self) -> bool:
        return self._config.save

    @save.setter
    def save(self, value: bool):
        self._config.save = value

    @property
    def save_debug(self) -> bool:
        return self._config.save_debug

    @save_debug.setter
    def save_debug(self, value: bool):
        self._config.save_debug = value

    @property
    def keep_subdirs(self) -> bool:
        return self._config.keep_subdirs

    @keep_subdirs.setter
    def keep_subdirs(self, value: bool):
        self._config.keep_subdirs = value

    @property
    def near_src(self) -> bool:
        return self._config.near_src

    @near_src.setter
    def near_src(self, value: bool):
        self._config.near_src = value

    @property
    def exist_ok(self) -> bool:
        return self._config.exist_ok

    @exist_ok.setter
    def exist_ok(self, value: bool):
        self._config.exist_ok = value

    @property
    def verbose(self) -> bool:
        return self._config.verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._config.verbose = value

    # --- Retrieval ---
    @property
    def config_dir(self) -> Path:
        """Return the root directory of all configuration files in the current
        project.
        """
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        """Return the root directory of all datasets in the current project."""
        return self.root / "data"

    @property
    def run_dir(self) -> Path:
        """Return the root directory of all runs in the current project."""
        return self.root / "run"

    @property
    def model_dir(self) -> Path | None:
        return MODELS.get_model_dir(self.model)

    @property
    def config_files(self) -> list[Path]:
        model = self.model
        config_dir = self.config_dir
        config_files = config_dir.files(
            f"*{model}*.yaml", f"*{model}*.yml",
            recursive=True
        ) if config_dir else []

        model_dir = self.model_dir
        config_files += model_dir.files(
            f"*{model}*.yaml", f"*{model}*.yml",
            recursive=True
        ) if model_dir else []

        return config_files

    @property
    def weights_files(self) -> list[Path]:
        model = self.model
        run_dir = self.run_dir
        weights_files = run_dir.files(
            f"*{model}*.pt", f"*{model}*.pth",
            recursive=True
        ) if run_dir else []

        weights_files += ZOO_ROOT.files(
            f"*{model}*.pt", f"*{model}*.pth",
            recursive=True
        ) if ZOO_ROOT else []
        return weights_files

    def resolve_save_dir(
        self,
        dirname: str,
        subdirname: str = "",
        src_path: PathLike | None = None,
    ) -> Path:
        """Compute the saving directory for an output type based on the output
        directory and optional components.

        The path is computed in the following order:
            ``<output_dir>``/``<dirname>``/``<sub_dirname>``/

        Examples:
            >>> output_dir = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm"
            >>> dirname = "pred"
            >>> subdirname = ""
            >>> src_path = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/image/01.jpg"
            >>> save_dir = resolve_save_dir(output_dir, dirname, subdirname, src_path, False, False)
            >>> print(save_dir)
            >>> # /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm/pred
            >>> save_dir = resolve_save_dir(output_dir, dirname, subdirname, src_path, True, False)
            >>> print(save_dir)
            >>> # /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm/test/image
            >>> save_dir = resolve_save_dir(output_dir, dirname, subdirname, src_path, False, True)
            >>> print(save_dir)
            >>> # /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/pred
            >>> save_dir = resolve_save_dir(output_dir, dirname, subdirname, src_path, True, True)
            >>> print(save_dir)
            >>> # /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/image_zerodce

        Args:
            output_dir (PathLike): Path to the output directory.
            dirname (str): Directory name to append to the output path
                (e.g., 'pred', 'debug').
            subdirname (str): Subdirectory name to append to the output path
                (e.g., 'debug'/'mask'). Defaults to "".
            src_path (PathLike, optional): Source path to determine the subdirectory
                hierarchy. Defaults to None.
            keep_subdirs (bool, optional): If True, preserve the subdirectory
                structure of ``src_path`` relative to ``dirname``. Defaults to False.
            near_src (bool, optional): If True, change the ``output_dir`` to the
                same level as ``src_path``. Defaults to False.

        Returns:
            Path: Computed output directory path.
        """
        return resolve_save_dir(
            output_dir=self.output_dir,
            dirname=dirname,
            subdirname=subdirname,
            src_path=src_path,
            keep_subdirs=self.keep_subdirs,
            near_src=self.near_src,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    # --- Mutation ---
    def update_from_yaml(self, path: PathLike):
        """Update the current configuration with values from a YAML file."""
        # Normalize inputs
        path = Path(path).normalize()

        # Validate inputs
        if not path.has_ext(".yaml", ".yml", exist=True):
            raise TypeError(
                f"Expected 'path' to be a valid configuration file path, "
                f"but got {type(path).__name__}."
            )

        new_config = Box.from_yaml(filename=path)
        merged_config = merge_dicts(self._config, new_config)
        self._config = Box(merged_config)
        self.config_file = path

    def update_from_cli(self, value: dict):
        """Update the current configuration with values from CLI arguments."""
        for k, v in value.items():
            if v is None:
                continue
            if k == "arch":
                self.arch = v
            elif k == "model":
                self.model = v
            elif k == "weights":
                self.weights = v
            elif k == "config":
                self.config_file = v
            elif k == "data":
                # Prediction data
                self.data = v
            else:
                self._config[k] = v

    def update_from_dict(self, value: DictLike):
        """Update the current configuration with values from a dictionary."""
        merged_config = merge_dicts(self.config, value)
        self._config = Box(merged_config)

    # --- Transformation ---
    def prepare_train(self) -> Box:
        """Prepare the current configuration for training.

        We utilize the setters to resolve the attributes in the correct values
        and types; and in the order of dependencies (e.g., model before weights).
        """
        # Add additional attributes
        self._config.hostname = socket.gethostname()

        # 1. Resolve standalone attributes first
        # 1.1. Resolve root
        self.root = self.root
        if not self.root or not self.root.is_dir():
            log_error(
                f"Project root not found at: {self.root}.\n"
                f"Set to the current working directory: {Path.cwd()}."
            )
            self.root = Path.cwd()

        # 1.2. Resolve config file
        self.config_file = self.config_file

        # 1.3. Resolve task and run mode
        self.task = self.task
        self.mode = self.mode

        # 1.4. Resolve model's arch and name
        self.arch = self.arch
        self.model = self.model

        # 1.5. Resolve device
        if not isinstance(self.device, torch.device):
            self.device = sys_ctx.get_torch_device(self.device)

        # 2. Resolve attributes that depend on other attributes
        # 2.1. Resolve dataloaders
        if self.config.train_dataloader:
            data_dir = self.config.train_dataloader.dataset.root
            data_dir = Path(data_dir).normalize() if data_dir else None
            if not data_dir or not data_dir.is_dir():
                self.config.train_dataloader.dataset.root = self.data_dir

        if self.config.val_dataloader:
            data_dir = self.config.val_dataloader.dataset.root
            data_dir = Path(data_dir).normalize() if data_dir else None
            if not data_dir or not data_dir.is_dir():
                self.config.val_dataloader.dataset.root = self.data_dir

        # 2.2. Resolve experiment name
        if not self.exp_name:
            config_file = self.config_file
            if config_file and config_file.is_config_file():
                # If no experiment name is given, use the config file name
                self.exp_name = config_file.stem
            else:
                # Otherwise, use the model name and train dataset name
                data = self.config.train_dataloader.dataset.name
                self.exp_name = f"{self.model}_{data}"

        # 2.3. Resolve the output directory
        output_dir = Path(self.output_dir) if self.output_dir else None
        if not output_dir or not output_dir.is_dir():
            self.output_dir = resolve_output_dir(
                root=self.run_dir,
                dirname="train",
                arch=self.arch,
                model=self.model,
                data=self.exp_name
            )
        else:
            self.output_dir = output_dir.normalize()
        if not self.exist_ok and self.output_dir.is_dir():
            self.output_dir.rmdir(recursive=True)

        # 2.4. Resolve weights
        if self.weights:
            weights = resolve_weights_file(self.root, self.weights)
            self.weights = create_weights(weights)
        if self.finetune:
            finetune = resolve_weights_file(self.root, self.finetune)
            self.finetune = create_weights(finetune)

        # Return the updated configuration
        return self.config

    def prepare_predict(self) -> Box:
        """Prepare the current configuration for prediction.

        We utilize the setters to resolve the attributes in the correct values
        and types; and in the order of dependencies (e.g., model before weights).
        """
        # Add additional attributes
        self._config.hostname = socket.gethostname()

        # 1. Resolve standalone attributes first
        # 1.1. Resolve root
        self.root = self.root
        if not self.root or not self.root.is_dir():
            log_error(
                f"Project root not found at: {self.root}.\n"
                f"Set to the current working directory: {Path.cwd()}."
            )
            self.root = Path.cwd()

        # 1.2. Resolve config file
        self.config_file = self.config_file

        # 1.3. Resolve task and run mode
        self.task = self.task
        self.mode = self.mode

        # 1.4. Resolve model's arch and name
        self.arch = self.arch
        self.model = self.model

        # 1.5. Resolve device
        if not isinstance(self.device, torch.device):
            self.device = sys_ctx.get_torch_device(self.device)

        # 2. Resolve attributes that depend on other attributes
        # 2.1 Resolve data
        self.data = self.data

        # 2.2. Resolve experiment name
        if not self.exp_name:
            config_file = self.config_file
            if config_file and config_file.is_config_file():
                # If no experiment name is given, use the config file name
                self.exp_name = config_file.stem
            else:
                # Otherwise, use the model name and train dataset name
                self.exp_name = self.model

        # 2.3. Resolve the output directory
        output_dir = Path(self.output_dir) if self.output_dir else None
        if not output_dir or not output_dir.is_dir():
            self.output_dir = resolve_output_dir(
                root=self.run_dir,
                dirname="predict",
                arch=self.arch,
                model=self.model
            )
        else:
            self.output_dir = output_dir.normalize()
        if not self.exist_ok and self.output_dir.is_dir():
            self.output_dir.rmdir(recursive=True)

        # 2.4. Resolve weights
        if self.weights:
            weights = resolve_weights_file(self.root, self.weights)
            self.weights = create_weights(weights)
        if self.finetune:
            finetune = resolve_weights_file(self.root, self.finetune)
            self.finetune = create_weights(finetune)

        # Return the updated configuration
        return self.config

    # --- Logging ---
    def log_summary(self, full: bool = False):
        """Log a summary of the current configuration for the current run.

        Args:
            full (bool, optional): If True, print the full configuration.
                Otherwise, print a concise summary. Defaults to False.
        """
        config = self._config.to_dict()

        if full:
            console.rule("[bold yellow]Full Configuration")
            # Ensure we have a standard dict for pretty printing
            pprint_dict(config)
        else:
            exp_name = config.get("exp_name", "Unnamed Run")
            console.rule(f"[bold red]{exp_name}")
            summary_fields = {
                "Machine": config.get("hostname", "local"),
                "Device": config.get("device"),
                "Task": config.get("task"),
                "Mode": config.get("mode"),
                "Data": config.get("data"),
                "Weights": config.get("weights"),
                "Save Dir": config.get("output_dir"),
                "Config": config.get("config"),
            }
            for label, value in summary_fields.items():
                if value:
                    # Formatting paths to be cleaner strings
                    display_val = str(value) if not isinstance(value, list) else f"{len(value)} files"
                    log(f"{label:<10}: {display_val}")
            console.rule() # Add a closing line for visual polish

# endregion


# ==============================================================================
# region CONTROL
# ==============================================================================

class ConfigContext(Config):
    """A class for managing configuration and performing run-time interactive
    prompting to update the configuration.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: PathLike,
        config: Box | None = None,
        config_file: PathLike | None = None,
        prompt: bool = False,
        **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (PathLike): Project root directory.
            config (Box, optional): Initial configuration to override the
                default config. Defaults to None.
            config_file (PathLike, optional): Path to the configuration file.
                If given, it will be loaded and used to override the default
                configuration. Defaults to None.
            prompt (bool, optional): If True, enable interactive prompting.
                Defaults to False.
            **kwargs: Additional keyword arguments for configuration updates.

        Raises:
            FileNotFoundError: If the project root directory is not found.
        """
        # Validate inputs
        if not root.is_dir():
            raise FileNotFoundError(f"Project root not found at: {root}")

        # Continue the initialization chain
        super().__init__(config=config, config_file=config_file, root=root, **kwargs)

        # Assign attributes
        self._index = 0

        # If prompting is enabled, run the interactive menu
        self.prompt() if prompt else None

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the total number of interactive steps."""
        return 17

    # --- Creation ---
    @classmethod
    def from_cli(cls, name: str = "main") -> "ConfigContext":
        """Create a new instance from CLI arguments."""
        # 1. Prepare argument parser
        parser = argparse.ArgumentParser(description=name)
        parser.add_argument("--prompt", "--p", action="store_true", help="Enable interactive prompting.")

        for opt_name, opt_params in ARGUMENTS.items():
            if opt_params.get("prompt_only", False):
                continue

            action = opt_params.get("action", "store")
            kwargs = {
                "action": action,
                "help": opt_params.get("help", ""),
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

        # 2. Parse CLI arguments
        args = parser.parse_args()
        args = vars(args)

        # 3. Create a new instance
        prompt = args.pop("prompt")
        root = args.pop("root") or Path.cwd()
        config_file = args.pop("config")
        return cls(root=root, config_file=config_file, prompt=prompt, **args)

    # --- Retrieval ---
    def config_for(self, mode: RunModeLike, prompt: bool = False) -> Box:
        """Get the resolved configuration for a specific run mode, optionally
        enabling interactive prompting.

        Args:
            mode (RunModeLike): The run mode for which to retrieve the configuration.
            prompt (bool, optional): If True, enable interactive prompting before
                returning the configuration. Defaults to False.

        Returns:
            Box: The resolved configuration for the specified run mode.
        """
        # Normalize inputs
        mode = RunMode(mode)

        # Update run mode
        self.mode = mode

        # If prompting is enabled, run the interactive menu to update the config
        self.prompt() if prompt else None

        # Return the resolved configuration based on the run mode
        if mode in [RunMode.TRAIN]:
            return self.prepare_train()
        elif mode in [RunMode.PREDICT]:
            return self.prepare_predict()
        else:
            raise ValueError(f"Invalid run mode: {mode}")

    # --- Transformation ---
    def as_config(self) -> Config:
        """Extracts the data and returns a pure ``Config`` object."""
        return Config(copy.deepcopy(self.config))

    # --- Prompting ---
    def prompt(self) -> Box:
        """Run the interactive menu until completion."""
        while True:
            self._display_prompt()
            if self._index == self.__len__():
                return self.config
            self._next()

    def _display_prompt(self):
        """Display the current prompt."""
        if self._index == 0:
            # clear_terminal()
            console.rule(f"[bold red]Input Prompts")
        else:
            console.rule()

        if self._index == 0:
            # Task
            self.task = OptionPrompt.ask(
                prompt=ARGUMENTS.task.prompt_text,
                default=self.task,
                choices=ARGUMENTS.task.choices,
            )
            # print(f"Task: {self.task}")
        if self._index == 1:
            # Mode
            self.mode = OptionPrompt.ask(
                prompt=ARGUMENTS.mode.prompt_text,
                default=self.mode,
                choices=ARGUMENTS.mode.choices,
            )
            # print(f"Mode: {self.mode}")
        if self._index == 2:
            # Arch
            self.arch = OptionPrompt.ask(
                prompt=ARGUMENTS.arch.prompt_text,
                default=self.arch,
                choices=MODELS.search_archs(self.task)
            )
            # print(f"Architecture: {self.arch}")
        if self._index == 3:
            # Model
            self.model = OptionPrompt.ask(
                prompt=ARGUMENTS.model.prompt_text,
                default=self.model,
                choices=MODELS.search(self.arch, self.task)
            )
            # print(f"Model: {self.model}")
        if self._index == 4:
            # Config file
            self.config_file = PathPrompt.ask(
                prompt=ARGUMENTS.config.prompt_text,
                default=self.config,
                choices=self.config_files,
                truncate_length=60,
                truncate_side="middle",
                commonpath=self.root,
                allow_empty=True,
            )
            # print(f"Config file: {self.config_file}")
            # print(f"{self.config}")
            self.update_from_yaml(self.config_file)  # First, update from a new config file
            self.update_from_cli(self.cli_kwargs)    # Then, add back CLI arguments
            # print(f"{self.config}")
        if self._index == 5:
            # Weights
            self.weights = PathPrompt.ask(
                prompt=ARGUMENTS.weights.prompt_text,
                default=resolve_weights_file(self.root, self.weights),
                choices=self.weights_files,
                truncate_length=60,
                truncate_side="middle",
                commonpath=self.root,
                allow_empty=True,
            )
            print(f"Weights: {self.weights}")
        if self._index == 6:
            # Data
            if self.mode not in [RunMode.PREDICT]:
                self._next()
            else:
                self.data = OptionPrompt.ask(
                    prompt=ARGUMENTS.data.prompt_text,
                    default=self.data,
                    choices=DATASETS.search(self.task, self.mode),
                    multiselect=True,
                    allow_empty=True,
                )
                # print(f"Data: {self.data}")
        if self._index == 7:
            # Experiment Name
            self.exp_name = Prompt.ask(
                prompt=ARGUMENTS.exp_name.prompt_text,
                default=self.exp_name,
            )
            # print(f"Experiment Name: {self.exp_name}")
        if self._index == 8:
            # Device
            self.device = OptionPrompt.ask(
                prompt=ARGUMENTS.device.prompt_text,
                default=sys_ctx.get_device(self.device).name,
                choices=ARGUMENTS.device.choices,
            )
            # print(f"Device: {self.device}")
        if self._index == 9:
            # Benchmark
            self.benchmark = Confirm.ask(
                prompt=ARGUMENTS.benchmark.prompt_text,
                default=self.config.benchmark,
            )
        if self._index == 10:
            # Save
            self.save = Confirm.ask(
                prompt=ARGUMENTS.save.prompt_text,
                default=self.config.save,
            )
        if self._index == 11:
            # Save Debug
            self.save_debug = Confirm.ask(
                prompt=ARGUMENTS.save_debug.prompt_text,
                default=self.config.save_debug,
            )
        if self._index == 12:
            # Keep Subdirs
            self.keep_subdirs = Confirm.ask(
                prompt=ARGUMENTS.keep_subdirs.prompt_text,
                default=self.config.keep_subdirs,
            )
        if self._index == 13:
            # Near Source
            self.near_src = Confirm.ask(
                prompt=ARGUMENTS.near_src.prompt_text,
                default=self.config.near_src,
            )
        if self._index == 14:
            # Exist OK
            self.exist_ok = Confirm.ask(
                prompt=ARGUMENTS.exist_ok.prompt_text,
                default=self.config.exist_ok,
            )
        if self._index == 15:
            # Verbose
            self.verbose = Confirm.ask(
                prompt=ARGUMENTS.verbose.prompt_text,
                default=self.config.verbose,
            )
        if self._index == 16:
            # Finish
            pprint_dict(self.config, title="Input Arguments")
            finish = Confirm.ask(prompt="Finish/Re-input", default=True)
            if finish:
                self._index = self.__len__()

    def _next(self):
        """Advance the prompt index by one."""
        self._index = (self._index + 1) % self.__len__()

    def _prev(self):
        """Move the prompt index back by one."""
        self._index = (self._index - 1) % self.__len__()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    ConfigContext(root=Path.cwd(), prompt=True)


# endregion
