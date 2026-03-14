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
from rich.table import Table

from mon.core.console import console, log_error, pprint_dict
from mon.core.constants import K
from mon.core.context import sys_ctx
from mon.core.data import Size, SizeLike, Weights
from mon.core.dtype import RunMode, Task
from mon.core.factory import DATASETS, MODELS, WEIGHTS
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
from mon.core.ui import Confirm, OptionPrompt, PathPrompt, Prompt
from mon.core.utils import is_valid_str, merge_dicts, truncate_string


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

    _DEFAULT_SCHEMA: Box = Box({
        # --- General ---
        "hostname": "localhost",
        "config_file": None,

        "exp_name": "",
        "root": None,
        "output_dir": None,
        "task": "",
        "mode": "",
        "device": torch.device("cpu"),
        "seed": 0,

        # --- Model ---
        "model": {
            "name": "",
            "arch": "",
            "weights": Weights(path=None),
            "finetune": Weights(path=None),
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
            "name": None,
            "lr": 0.0,
            "weight_decay": 0.0,
        },
        "lr_scheduler": {},
        "lr_warmup_scheduler": {},
        "loss": {},
        "tensorboard_logger": True,

        # --- Prediction ---
        "eval_imgsz": None,
        "benchmark": False,

        # --- Saving & Visualization ---
        "save": True,
        "save_debug": False,
        "keep_subdirs": False,
        "near_src": False,
        "exist_ok": True,
        "verbose": True,
    })

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
        self._config: Box = copy.deepcopy(self._DEFAULT_SCHEMA)
        self.infer_data = None

        # Assign attributes
        self.root = root
        self.config_file = config_file
        self.cli_kwargs = kwargs

        # Update the config with the given values
        if config:
            self.update_from_dict(config)
        if self.config_file:
            # If a config file is given, load it and update the default config
            self.update_from_yaml(self.config_file)
        if self.cli_kwargs:
            # If arguments are given from CLI, update the config
            self.update_from_cli(self.cli_kwargs)

    # --- Attribute Access ---
    def __getattr__(self, name: str) -> Any:
        """Called only if the attribute was not found in the usual places.

        Triggered ONLY when an attribute is NOT found via normal lookup
        (i.e., it's not a @property and not a standard attribute).
        We catch it here and forward it to the internal Box.
        """
        # Guard against Python's internal checks before ``self._config`` is
        # initialized
        if "_config" not in self.__dict__:
            raise AttributeError(name)

        # Delegate the lookup to the internal Box
        try:
            return getattr(self._config, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """Intercept every attribute assignment.

        Intercepts ALL assignments to route data correctly between the wrapper
        class and the internal Box.
        """
        # 1. Is it a defined @property? Let the base class trigger your setter.
        if isinstance(getattr(self.__class__, name, None), property):
            super().__setattr__(name, value)
            return

        # 2. Are we still inside __init__? Let the standard assignment happen.
        # (This prevents infinite loops when setting ``self._config`` the first
        # time)
        if name == "_config" or "_config" not in self.__dict__:
            super().__setattr__(name, value)
            return

        # 3. Otherwise, inject it directly into the inner Box!
        self._config[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with the given ``key`` from the configuration.

        This is an alias to ``dict.get(key, default)``.
        """
        return self._config.get(key, default)

    # --- Properties ---
    @property
    def config(self) -> Box:
        """Return the internal configuration dictionary."""
        return self._config

    @property
    def config_file(self) -> Path | None:
        """Return the path to the configuration file."""
        return self.config.config_file

    @config_file.setter
    def config_file(self, value: PathLike | None):
        """Set the path to the configuration file."""
        # Validate inputs
        if not is_valid_str(value):
            return

        # Check if the given value is a valid path
        config_file = Path(value).normalize()
        if config_file.has_ext(".yaml", ".yml", exists=True):
            self._config.config_file = config_file
            return

        # Look for the configuration file in the project config directory
        if self.config_dir:
            config_file = self.config_dir / value
            if config_file.has_ext(".yaml", ".yml", exists=True):
                self._config.config_file = config_file
                return

    @property
    def exp_name(self) -> str:
        """Return the experiment name."""
        if self.config.exp_name:
            return self.config.exp_name
        elif self.config_file:
            return self.config_file.stem
        else:
            return self.model_name

    @exp_name.setter
    def exp_name(self, value: str):
        """Set the experiment name."""
        self._config.exp_name = value

    @property
    def root(self) -> Path:
        """Return the project root directory."""
        return self.config.root

    @root.setter
    def root(self, value: PathLike | None):
        """Set the project root directory."""
        root = Path(value).normalize() if is_valid_str(value) else None
        if root and root.is_dir():
            # If the given value is a valid directory, set it as the root
            self._config.root = root
        # else:
            # self._config.root = resolve_project_root(Path.cwd())

    @property
    def output_dir(self) -> Path | None:
        """Return the output directory."""
        return self._config.output_dir

    @output_dir.setter
    def output_dir(self, value: PathLike | None):
        """Set the output directory."""
        output_dir = Path(value).normalize() if is_valid_str(value) else None
        if output_dir:  # and output_dir.is_dir():
            self._config.output_dir = output_dir

    @property
    def task(self) -> Task | None:
        """Return the task type."""
        return self._config.task

    @task.setter
    def task(self, value: TaskLike | None):
        """Set the task type."""
        if value in Task:
            self._config.task = Task(value)

    @property
    def mode(self) -> RunMode | None:
        """Return the run mode."""
        return self._config.mode

    @mode.setter
    def mode(self, value: RunModeLike | None):
        """Set the run mode."""
        if value in RunMode:
            self._config.mode = RunMode(value)

    @property
    def arch(self) -> str:
        """Return the model architecture."""
        return self._config.model.arch

    @arch.setter
    def arch(self, value: str | None):
        """Set the model architecture."""
        if is_valid_str(value):
            self._config.model.arch = value

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._config.model.name

    @model_name.setter
    def model_name(self, value: str | None):
        """Set the model name."""
        if is_valid_str(value):
            self._config.model.name = value

    @property
    def weights(self) -> Weights | None:
        """Return the model weights."""
        return self._config.model.weights

    @weights.setter
    def weights(self, value: Weights | PathLike | None):
        """Set the model weights."""
        if isinstance(value, Weights):
            # If the value is already a Weights object, set it directly
            self._config.model.weights = value
        elif WEIGHTS.find_weights_obj(weights_path=value):
            # If the value matches a registered weights object in WEIGHTS, use it
            self._config.model.weights = WEIGHTS.find_weights_obj(weights_path=value)
        elif is_valid_str(value):
            # If the value is a valid string, treat it as a path and create a
            # Weights object
            self._config.model.weights = Weights(path=value)

    @property
    def finetune(self) -> Weights | None:
        """Return the model weights."""
        return self._config.model.finetune

    @finetune.setter
    def finetune(self, value: Weights | PathLike | None):
        """Set the model weights."""
        if isinstance(value, Weights):
            # If the value is already a Weights object, set it directly
            self._config.model.finetune = value
        elif WEIGHTS.find_weights_obj(weights_path=value):
            # If the value matches a registered weights object in WEIGHTS, use it
            self._config.model.finetune = WEIGHTS.find_weights_obj(weights_path=value)
        elif is_valid_str(value):
            # If the value is a valid string, treat it as a path and create a
            # Weights object
            self._config.model.finetune = Weights(path=value)

    @property
    def data(self) -> list[PathLike]:
        """Return the list of inference data sources."""
        return self._config.data

    @data.setter
    def data(self, value: list[PathLike] | PathLike | None):
        """Set the list of inference data sources."""
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
        """Return the device to use for computation."""
        return self._config.device

    @device.setter
    def device(self, value: DeviceLike):
        """Set the device to use for computation."""
        self._config.device = sys_ctx.get_torch_device(value)

    @property
    def eval_imgsz(self) -> Size | None:
        """Return the evaluation image size."""
        return self._config.eval_imgsz

    @eval_imgsz.setter
    def eval_imgsz(self, value: SizeLike | None):
        """Set the evaluation image size."""
        if value is not None:
            self._config.eval_imgsz = Size.from_value(value)

    @property
    def benchmark(self) -> bool:
        """Return whether to run in benchmark mode."""
        return self._config.benchmark

    @benchmark.setter
    def benchmark(self, value: bool):
        """Set whether to run in benchmark mode."""
        self._config.benchmark = value

    @property
    def save(self) -> bool:
        """Return whether to save the output."""
        return self._config.save

    @save.setter
    def save(self, value: bool):
        """Set whether to save the output."""
        self._config.save = value

    @property
    def save_debug(self) -> bool:
        """Return whether to save debug information."""
        return self._config.save_debug

    @save_debug.setter
    def save_debug(self, value: bool):
        """Set whether to save debug information."""
        self._config.save_debug = value

    @property
    def keep_subdirs(self) -> bool:
        """Return whether to keep subdirectories in the output directory."""
        return self._config.keep_subdirs

    @keep_subdirs.setter
    def keep_subdirs(self, value: bool):
        """Set whether to keep subdirectories in the output directory."""
        self._config.keep_subdirs = value

    @property
    def near_src(self) -> bool:
        """Return whether to keep subdirectories in the output directory."""
        return self._config.near_src

    @near_src.setter
    def near_src(self, value: bool):
        """Set whether to keep subdirectories in the output directory."""
        self._config.near_src = value

    @property
    def exist_ok(self) -> bool:
        """Return whether to overwrite existing files."""
        return self._config.exist_ok

    @exist_ok.setter
    def exist_ok(self, value: bool):
        """Set whether to overwrite existing files."""
        self._config.exist_ok = value

    @property
    def verbose(self) -> bool:
        """Return whether to enable verbose mode."""
        return self._config.verbose

    @verbose.setter
    def verbose(self, value: bool):
        """Set whether to enable verbose mode."""
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
        """Return the root directory of all models in the current project."""
        return MODELS.get_model_dir(self.model_name)

    @property
    def config_files(self) -> list[Path]:
        """Return a list of all configuration files in the current project."""
        model = self.model_name
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

        return sorted(config_files)

    @property
    def weights_files(self) -> list[Path]:
        """Return a list of all weights files in the current project."""
        model = self.model_name
        run_dir = self.run_dir

        # 1. Look for weights files in the current run directory
        weights_files = run_dir.files(
            f"*{model}*.pt",
            f"*{model}*.pth",
            f"*/*{model}*/*.pt",
            f"*/*{model}*/*.pth",
            recursive=True
        ) if run_dir else []

        # 2. Look for weights files in the global zoo directory
        weights_files += K.ZOO_ROOT.files(
            f"*{model}*.pt",
            f"*{model}*.pth",
            recursive=True
        ) if K.ZOO_ROOT else []

        return sorted(weights_files)

    @property
    def infer_data(self) -> PathLike:
        """Return the current inference data source."""
        return self._current_data or ""

    @infer_data.setter
    def infer_data(self, value: PathLike | None):
        """Set the current inference data source."""
        self._current_data = Path(value).normalize() if is_valid_str(value) else None

    @property
    def infer_data_name(self) -> str:
        """Return the current inference data source name."""
        if self._current_data is None:
            return ""
        if self._current_data.is_file():
            return self._current_data.stem
        else:
            return self._current_data.name

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
            dirname (str): Directory name to append to the output path
                (e.g., 'pred', 'debug').
            subdirname (str): Subdirectory name to append to the output path
                (e.g., 'debug'/'mask'). Defaults to "".
            src_path (PathLike, optional): Source path to determine the subdirectory
                hierarchy. Defaults to None.

        Returns:
            Path: Computed output directory path.
        """
        return resolve_save_dir(
            output_dir=self.output_dir / self.infer_data_name,
            dirname=dirname,
            subdirname=subdirname,
            src_path=src_path,
            keep_subdirs=self.keep_subdirs,
            near_src=self.near_src,
        )

    def resolve_save_file(
        self,
        dirname: str,
        src_path: PathLike,
        subdirname: str = "",
    ) -> Path:
        """Compute the saving file path for an output type based on the output
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
            dirname (str): Directory name to append to the output path
                (e.g., 'pred', 'debug').
            src_path (PathLike): Source path to determine the subdirectory
                hierarchy.
            subdirname (str): Subdirectory name to append to the output path
                (e.g., 'debug'/'mask'). Defaults to "".

        Returns:
            Path: Computed output directory path.
        """
        # Normalize inputs
        src_path = Path(src_path).normalize()

        save_dir = resolve_save_dir(
            output_dir=self.output_dir / self.infer_data_name,
            dirname=dirname,
            subdirname=subdirname,
            src_path=src_path,
            keep_subdirs=self.keep_subdirs,
            near_src=self.near_src,
        )
        return save_dir / src_path.name

    # --- Mutation ---
    def update_from_yaml(self, path: PathLike):
        """Update the current configuration with values from a YAML file."""
        # Normalize inputs
        path = Path(path).normalize()

        # Validate inputs
        if not path.has_ext(".yaml", ".yml", exists=True):
            raise TypeError(
                f"Expected 'path' to be a valid configuration file path, "
                f"but got {type(path).__name__}."
            )

        new_config = Box.from_yaml(filename=path)
        # merged_config = merge_dicts(self._config, new_config)
        # self._config = Box(merged_config)
        self.update_from_dict(new_config)
        self.config_file = path

    def update_from_cli(self, value: dict):
        """Update the current configuration with values from CLI arguments."""
        for k, v in value.items():
            if (
                (k in ARGUMENTS and v == ARGUMENTS[k].get("default"))
                or v is None
                or (isinstance(v, (list, tuple, dict)) and len(v) == 0)
            ):
                continue

            if k == "arch":
                self.arch = v
            elif k == "model":
                self.model_name = v
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
        for key, val in value.items():
            if val is None:
                continue

            # Check if this key has a dedicated @property setter in this class
            if hasattr(self.__class__, key) and isinstance(getattr(self.__class__, key), property):
                # This perfectly triggers your validation! (e.g., self.device = val)
                setattr(self, key, val)
            else:
                # If no setter exists, just update the Box dynamically
                if isinstance(val, dict) and key in self._config:
                    # self._config[key].merge_update(val) # Box's built-in deep merge
                    self._config[key] = Box(merge_dicts(self._config[key], val))
                else:
                    self._config[key] = val

        self._force_validation(
            "task", "mode", "arch", "model", "weights", "finetune"
        )

    def prepare_for_train(self):
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

        # 1.2. Clean, explicit, and lint-friendly!
        self._force_validation(
            "config_file", "task", "mode", "arch", "model", "eval_imgsz"
        )

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
                self.exp_name = f"{self.model_name}_{data}"

        # 2.3. Resolve the output directory
        output_dir = Path(self.output_dir) if self.output_dir else None
        if not output_dir or not output_dir.is_dir():
            self.output_dir = resolve_output_dir(
                root=self.run_dir,
                dirname="train",
                arch=self.arch,
                # model=self.model_name,
                data=self.exp_name,
            )
        else:
            self.output_dir = output_dir.normalize()
        if not self.exist_ok and self.output_dir.is_dir():
            self.output_dir.rmdir(recursive=True)

        # 2.4. Resolve weights
        if self.weights:
            # weights = resolve_weights_file(self.root, self.weights.path)
            # self.weights = create_weights(weights)
            self.weights.rectify_path(root=self.root)
        if self.finetune:
            # finetune = resolve_weights_file(self.root, self.finetune.path)
            # self.finetune = create_weights(finetune)
            self.finetune.rectify_path(root=self.root)

    def prepare_for_predict(self):
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

        # 1.2. Clean, explicit, and lint-friendly!
        self._force_validation(
            "config_file", "task", "mode", "arch", "model", "eval_imgsz"
        )

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
                self.exp_name = self.model_name

        # 2.3. Resolve the output directory
        output_dir = Path(self.output_dir) if self.output_dir else None
        if not output_dir or not output_dir.is_dir():
            self.output_dir = resolve_output_dir(
                root=self.run_dir,
                dirname="predict",
                arch=self.arch,
                # model=self.model_name,
                model=self.exp_name,
            )
        else:
            self.output_dir = output_dir.normalize()
        if not self.exist_ok and self.output_dir.is_dir():
            self.output_dir.rmdir(recursive=True)

        # 2.4. Resolve weights
        if self.weights:
            self.weights.rectify_path(root=self.root)
        if self.finetune:
            self.finetune.rectify_path(root=self.root)

    def _force_validation(self, *properties):
        """Forces existing config values to pass through their property setters.

        Args:
            *properties: The names of the @property attributes to validate
                (e.g., 'root', 'arch', 'model_name').
        """
        for prop in properties:
            # Check if this string actually corresponds to a @property on the class
            if (
                hasattr(self.__class__, prop)
                and isinstance(getattr(self.__class__, prop), property)
            ):
                # getattr(self, prop) uses your custom getter to find the value (even nested ones!)
                # setattr(self, prop, ...) routes it through your custom setter for validation
                current_value = getattr(self, prop)
                setattr(self, prop, current_value)

    # --- Logging ---
    def log_summary(self, full: bool = False):
        """Log a summary of the current configuration for the current run.

        Args:
            full (bool, optional): If True, print the full configuration.
                Otherwise, print a concise summary. Defaults to False.
        """
        if full:
            console.rule("[bold yellow]Full Configuration")
            # Ensure we have a standard dict for pretty printing
            pprint_dict(self.config.to_dict())
        else:
            console.rule(f"[bold red]{self.exp_name}")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Option", style="dim")
            table.add_column("Value", justify="left")
            #
            table.add_row("Machine", self.config.get("hostname", "local"))
            table.add_row("Device", str(self.device))
            table.add_row("Task", self.task)
            table.add_row("Mode", self.mode)
            table.add_row("Data", "\n".join([truncate_string(d, side="left") for d in self.data]))
            table.add_row("Weights", truncate_string(self.weights.path))
            table.add_row("Save Dir", truncate_string(self.output_dir))
            table.add_row("Config", truncate_string(self.config_file))
            #
            console.log(table)
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
    def from_cli(
        cls,
        root: PathLike | None = None,
        config_file: PathLike | None = None,
        name: str = "main",
        **kwargs
    ) -> "ConfigContext":
        """Create a new instance from CLI arguments.

        Args:
            root (PathLike | None): Optional project root directory.
                Defaults to None.
            config_file (PathLike | None): Optional path to a configuration file.
                Defaults to None.
            name (str): Name for the argument parser. Defaults to "main".

        Returns:
            ConfigContext: A new instance of ConfigContext initialized with CLI
                arguments.
        """
        # 1. Prepare argument parser
        parser = argparse.ArgumentParser(description=name)
        parser.add_argument("--prompt", "--p", action="store_true", help="Enable interactive prompting.")

        for opt_name, opt_params in ARGUMENTS.items():
            if opt_params.get("prompt_only", False):
                continue

            action = opt_params.get("action", "store")
            argument_kwargs = {
                "action": action,
                "help": opt_params.get("help", ""),
                "required": opt_params.get("required", False),
            }

            # Boolean actions (store_true/store_false) do not take 'type' or 'choices'
            if action in ["store_true", "store_false"]:
                # Default is usually False for store_true, True for store_false
                argument_kwargs["default"] = opt_params.get("default", action == "store_false")
            else:
                if "type" in opt_params:
                    argument_kwargs["type"] = opt_params["type"]
                if "choices" in opt_params:
                    argument_kwargs["choices"] = opt_params["choices"]
                argument_kwargs["default"] = opt_params.get("default", None)

            # If a default value for this argument is provided in kwargs, it
            # should override the default from ARGUMENTS
            if opt_name in kwargs:
                argument_kwargs["default"] = kwargs[opt_name]

            flag = f"--{opt_name.replace('_', '-')}"
            parser.add_argument(flag, **argument_kwargs)

        parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")

        # 2. Parse CLI arguments
        args = parser.parse_args()
        args = vars(args)

        # 3. Create a new instance
        prompt = args.pop("prompt")
        root = args.pop("root") or root or Path.cwd()
        config_file = args.pop("config") or config_file
        return cls(root=root, config_file=config_file, prompt=prompt, **args)

    # --- Retrieval ---
    def config_for(self, mode: RunModeLike, prompt: bool = False) -> Config:
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
            self.prepare_for_train()
        elif mode in [RunMode.PREDICT]:
            self.prepare_for_predict()
        else:
            raise ValueError(f"Invalid run mode: {mode}")

        return self  # self.as_config()

    # --- Transformation ---
    def as_config(self) -> Config:
        """Extracts the data and returns a pure ``Config`` object."""
        new_config = Config()
        new_config._config = self.config
        return new_config

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
        if self._index == 1:
            # Mode
            self.mode = OptionPrompt.ask(
                prompt=ARGUMENTS.mode.prompt_text,
                default=self.mode,
                choices=ARGUMENTS.mode.choices,
            )
        if self._index == 2:
            # Arch
            self.arch = OptionPrompt.ask(
                prompt=ARGUMENTS.arch.prompt_text,
                default=self.arch,
                choices=MODELS.search_archs(self.task)
            )
        if self._index == 3:
            # Model
            self.model_name = OptionPrompt.ask(
                prompt=ARGUMENTS.model.prompt_text,
                default=self.model_name,
                choices=MODELS.search(self.arch, self.task)
            )
        if self._index == 4:
            # Config file
            self.config_file = PathPrompt.ask(
                prompt=ARGUMENTS.config.prompt_text,
                default=self.config,
                choices=self.config_files,
                column_first=True,
                truncate_length=60,
                truncate_side="middle",
                commonpath=self.root,
                allow_empty=True,
            )
            # First, update from a new config file
            self.update_from_yaml(self.config_file)
            # Then, add back CLI arguments
            self.update_from_cli(self.cli_kwargs)
            """
            self.mode = self.cli_kwargs["mode"]
            self.save = self.cli_kwargs["save"]
            self.save_debug = self.cli_kwargs["save_debug"]
            self.keep_subdirs = self.cli_kwargs["keep_subdirs"]
            self.near_src = self.cli_kwargs["near_src"]
            self.exist_ok = self.cli_kwargs["exist_ok"]
            self.verbose = self.cli_kwargs["verbose"]
            """
        if self._index == 5:
            # Weights
            self.weights = PathPrompt.ask(
                prompt=ARGUMENTS.weights.prompt_text,
                default=resolve_weights_file(self.root, self.weights.path),
                choices=self.weights_files,
                column_first=True,
                truncate_length=60,
                truncate_side="middle",
                commonpath=self.root,
                allow_empty=True,
            )
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
        if self._index == 7:
            # Experiment Name
            self.exp_name = Prompt.ask(
                prompt=ARGUMENTS.exp_name.prompt_text,
                default=self.exp_name,
            )
        if self._index == 8:
            # Device
            self.device = OptionPrompt.ask(
                prompt=ARGUMENTS.device.prompt_text,
                default=sys_ctx.get_device(self.device).name,
                choices=ARGUMENTS.device.choices,
            )
        if self._index == 9:
            # Benchmark
            self.benchmark = Confirm.ask(
                prompt=ARGUMENTS.benchmark.prompt_text,
                default=self.benchmark,
            )
        if self._index == 10:
            # Save
            self.save = Confirm.ask(
                prompt=ARGUMENTS.save.prompt_text,
                default=self.save,
            )
        if self._index == 11:
            # Save Debug
            self.save_debug = Confirm.ask(
                prompt=ARGUMENTS.save_debug.prompt_text,
                default=self.save_debug,
            )
        if self._index == 12:
            # Keep Subdirs
            self.keep_subdirs = Confirm.ask(
                prompt=ARGUMENTS.keep_subdirs.prompt_text,
                default=self.keep_subdirs,
            )
        if self._index == 13:
            # Near Source
            self.near_src = Confirm.ask(
                prompt=ARGUMENTS.near_src.prompt_text,
                default=self.near_src,
            )
        if self._index == 14:
            # Exist OK
            self.exist_ok = Confirm.ask(
                prompt=ARGUMENTS.exist_ok.prompt_text,
                default=self.exist_ok,
            )
        if self._index == 15:
            # Verbose
            self.verbose = Confirm.ask(
                prompt=ARGUMENTS.verbose.prompt_text,
                default=self.verbose,
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
