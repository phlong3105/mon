#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config Data Structures & Management.

This module provides data structures and utilities for managing configuration
files.
"""

from __future__ import annotations

__all__ = [
    # "ARGUMENTS",
    "ConfigHandler",
    "ConfigManager",
    "create_default_config",
    "load_config",
]

import argparse
import copy
import socket
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

import torch
from box import Box

from mon.core.constants import DATASETS, MODELS, ZOO_ROOT
from mon.core.data import create_weights, DEVICE_MANAGER
from mon.core.dtype import RunMode, Task
from mon.core.filesystem import resolve_output_dir, resolve_weights_file
from mon.core.path import Path
from mon.core.typing import DeviceLike, PathLike, RunModeLike, TaskLike
from mon.core.ui import (
    Confirm,
    console,
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
        "choices": [None] + DEVICE_MANAGER.names,
        "help": f"Running device: {[None] + DEVICE_MANAGER.names}.",
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

# --- Data Structures ---

@dataclass
class ModelConfig:
    """Data structure for storing model configuration."""

    name: str = ""
    arch: str = ""
    weights: Path | None = None
    finetune: Path | None = None


@dataclass
class TransformConfig:
    """Data structure for storing transform configuration."""

    ops: list = field(default_factory=list)
    p: float = 1.0
    seed: int | None = None


@dataclass
class DatasetConfig:
    """Data structure for storing dataset configuration."""

    name: str = ""
    root: Path | None = None
    dirname: str = ""
    subdir: str = ""
    split: str = "train"
    transforms: TransformConfig = field(default_factory=TransformConfig)
    modalities: list = field(default_factory=list)
    classes: Path | None = None
    verbose: bool = True


@dataclass
class DataLoaderConfig:
    """Data structure for storing dataloader configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 4
    drop_last: bool = False


@dataclass
class OptimizerConfig:
    """Data structure for storing optimizer configuration."""

    name: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class ExperimentConfig:
    """Data structure for storing experiment configuration."""

    # --- General ---
    exp_name: str = ""
    root: Path | None = None
    output_dir: Path | None = None
    task: str = ""
    mode: str = ""
    device: str | int = 0
    seed: int = 0

    # --- Model ---
    model: ModelConfig = field(default_factory=ModelConfig)

    # --- Data ---
    train_dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    val_dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    # --- Training ---
    epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: dict = field(default_factory=dict)
    lr_warmup_scheduler: dict = field(default_factory=dict)

    # --- Prediction ---
    benchmark: bool = False

    # --- Saving & Visualization ---
    save: bool = True
    save_debug: bool = False
    keep_subdirs: bool = False
    near_src: bool = False
    exist_ok: bool = True
    verbose: bool = True

# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

def load_config(path: PathLike, **kwargs: Any) -> Box:
    """Load a configuration file and return a Box-like dictionary."""
    # Normalize inputs
    path = Path(path).normalize()

    # Validate inputs
    if not path.has_ext(".yaml", ".yml", exist=True):
        raise TypeError(
            f"Expected 'path' to be a valid configuration file path, "
            f"but got {type(path).__name__}."
        )

    # Create a default configuration
    config = Box(asdict(ExperimentConfig()))
    # Update with .yaml file contents
    config.update(Box.from_yaml(filename=path), **kwargs)
    return config

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_default_config() -> Box:
    """Create a default configuration Box-like dictionary."""
    return Box(asdict(ExperimentConfig()))

# endregion


# ==============================================================================
# region CONTROL
# ==============================================================================

class ConfigHandler:
    """A helper class for managing configuration access and updates."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        config: Box | None = None,
        config_file: PathLike | None = None,
        root: PathLike | None = None,
        **kwargs
    ):
        """Initialize a new instance.

        The configuration initialization pipeline is as follows:
            ``default`` -> ``config`` -> ``config_file`` -> ``kwargs``

        Args:
            config (Box, optional): Initial configuration to override the
                default config. Defaults to None.
            config_file (PathLike, optional): Path to the configuration file.
                If given, it will be loaded and used to override the default
                configuration. Defaults to None.
            root (PathLike): Project root directory.
            **kwargs: Additional keyword arguments for configuration updates.
        """
        # Allocate resources
        # Create a default configuration
        self.config = create_default_config()
        if config:
            # If an initial configuration is given, update the default config
            self.update_from_config(config)

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
    def config_file(self) -> Path | None:
        """Return the path to the configuration file."""
        return self.config.get("config", None)

    @config_file.setter
    def config_file(self, value: PathLike | None):
        """Set the path to the configuration file."""
        # Validate inputs
        if not is_valid_str(value):
            self.config["config"] = None
            return

        # Check if the given value is a valid path
        config_file = Path(value).normalize()
        if config_file.has_ext(".yaml", ".yml", exist=True):
            self.config["config"] = config_file
            return

        # Look for the configuration file in the project config directory
        if self.config_dir:
            config_file = self.config_dir / value
            if config_file.has_ext(".yaml", ".yml", exist=True):
                self.config["config"] = config_file
                return

    @property
    def exp_name(self) -> str | None:
        """Return the experiment name."""
        if self.config.exp_name:
            return self.config.exp_name
        elif self.config_file:
            return self.config_file.stem
        elif self.data:
            return f"{self.model}_{self.data}"
        else:
            return self.model

    @exp_name.setter
    def exp_name(self, value: str | None):
        """Set the experiment name."""
        self.config.exp_name = value

    @property
    def root(self) -> Path:
        """Return the project root directory."""
        return self.config.root

    @root.setter
    def root(self, value: PathLike | None):
        """Set the project root directory."""
        # root = resolve_project_root(value)
        root = Path(value).normalize() if value else None
        if root and root.is_dir():
           self.config.root = root

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
    def task(self) -> Task | None:
        return self.config.task

    @task.setter
    def task(self, value: TaskLike | None):
        self.config.task = Task(value) if value else None

    @property
    def mode(self) -> RunMode | None:
        return self.config.mode

    @mode.setter
    def mode(self, value: RunModeLike | None):
        self.config.mode = RunMode(value) if value else None

    @property
    def arch(self) -> str | None:
        return self.config.model.arch

    @arch.setter
    def arch(self, value: str | None):
        self.config.model.arch = value

    @property
    def model(self) -> str | None:
        return self.config.model.name

    @model.setter
    def model(self, value: str | None):
        self.config.model.name = value

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
    def weights(self) -> Path | None:
        return self.config.model.weights

    @weights.setter
    def weights(self, value: PathLike | None):
        self.config.model.weights = Path(value) if value else None

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

    @property
    def data(self) -> str | None:
        return self.config.get("data")

    @data.setter
    def data(self, value: str | None):
        self.config["data"] = value

    @property
    def device(self) -> DeviceLike:
        return self.config.device

    @device.setter
    def device(self, value: DeviceLike):
        self.config.device = DEVICE_MANAGER.get_device(value)

    @property
    def benchmark(self) -> bool:
        return self.config.benchmark

    @benchmark.setter
    def benchmark(self, value: bool):
        self.config.benchmark = value

    @property
    def save(self) -> bool:
        return self.config.save

    @save.setter
    def save(self, value: bool):
        self.config.save = value

    @property
    def save_debug(self) -> bool:
        return self.config.save_debug

    @save_debug.setter
    def save_debug(self, value: bool):
        self.config.save_debug = value

    @property
    def keep_subdirs(self) -> bool:
        return self.config.keep_subdirs

    @keep_subdirs.setter
    def keep_subdirs(self, value: bool):
        self.config.keep_subdirs = value

    @property
    def near_src(self) -> bool:
        return self.config.near_src

    @near_src.setter
    def near_src(self, value: bool):
        self.config.near_src = value

    @property
    def exist_ok(self) -> bool:
        return self.config.exist_ok

    @exist_ok.setter
    def exist_ok(self, value: bool):
        self.config.exist_ok = value

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @verbose.setter
    def verbose(self, value: bool):
        self.config.verbose = value

    # --- Mutation ---
    def update_from_yaml(self, path: PathLike):
        """Update the current configuration with values from a YAML file."""
        loaded_config = load_config(path)
        merged_config = merge_dicts(self.config, loaded_config)
        self.config = Box(merged_config)
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
                self.config["data"] = v
            else:
                self.config[k] = v

    def update_from_config(self, value: Box):
        """Update the current configuration with values from another config Box."""
        merged_config = merge_dicts(self.config, value)
        self.config = Box(merged_config)


class ConfigManager(ConfigHandler):
    """A class for managing configuration and performing interactive prompts."""

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
    def from_cli(cls, name: str = "main") -> "ConfigManager":
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
    def train(self, prompt: bool = False) -> Box:
        """Resolve the current configuration for a training run.

        Args:
            prompt (bool, optional): If True, enable interactive prompting
                before returning the configuration. Defaults to False.

        Returns:
            Box: The resolved configuration for training.
        """
        # If prompting is enabled, run the interactive menu to update the config
        self.prompt() if prompt else None

        # Retrieve attributes
        config = copy.deepcopy(self.config)
        config_file = self.config_file

        # Add additional attributes
        config["hostname"] = socket.gethostname()
        config["config"] = config_file

        # Resolve experiment name
        if not config.exp_name:
            if config_file.is_config_file():
                # If no experiment name is given, use the config file name
                config.exp_name = config_file.stem
            else:
                # Otherwise, use the model name and train dataset name
                model = config.model.name
                data = config.train_dataloader.dataset.name
                config.exp_name = f"{model}_{data}"

        # Update project root
        config.root = self.root

        # Resolve the output directory
        output_dir = Path(config.output_dir) if config.output_dir else None
        if not output_dir.is_dir():
            config.output_dir = resolve_output_dir(
                root=self.run_dir,
                dirname="train",
                arch=config.model.arch,
                model=config.model.name,
                data=config.exp_name
            )
        else:
            config.output_dir = output_dir.normalize()
        if not config.exist_ok and config.output_dir.is_dir():
            config.output_dir.rmdir(recursive=True)

        # Resolve device
        if not isinstance(config.device, torch.device):
            device = config.device
            config.device = DEVICE_MANAGER.get_device(device)

        # Resolve weights
        if config.model.weights:
            weights = config.model.weights
            weights = resolve_weights_file(self.root, weights)
            config.model.weights = create_weights(weights)

        # Return the updated configuration
        return config

    def predict(self, prompt: bool = False) -> Box:
        """Resolve the current configuration for a prediction run.

        Args:
            prompt (bool, optional): If True, enable interactive prompting
                before returning the configuration. Defaults to False.

        Returns:
            Box: The resolved configuration for prediction.
        """
        # If prompting is enabled, run the interactive menu to update the config
        self.prompt() if prompt else None

        # Retrieve attributes
        config = copy.deepcopy(self.config)
        config_file = self.config_file

        # Add additional attributes
        config["hostname"] = socket.gethostname()
        config["config"] = config_file

        # Resolve experiment name
        if not config.exp_name:
            if config_file.is_config_file():
                # If no experiment name is given, use the config file name
                config.exp_name = config_file.stem
            else:
                # Otherwise, use the model name and train dataset name
                model = config.model.name
                data = config.train_dataloader.dataset.name
                config.exp_name = f"{model}_{data}"

        # Update project root
        config.root = self.root

        # Resolve the output directory
        output_dir = Path(config.output_dir) if config.output_dir else None
        if not output_dir.is_dir():
            config.output_dir = resolve_output_dir(
                root=self.run_dir,
                dirname="predict",
                arch=config.model.arch,
                model=config.model.name,
            )
        else:
            config.output_dir = output_dir.normalize()
        if not config.exist_ok and config.output_dir.is_dir():
            config.output_dir.rmdir(recursive=True)

        # Resolve device
        if not isinstance(config.device, torch.device):
            device = config.device
            config.device = DEVICE_MANAGER.get_device(device)

        # Return the updated configuration
        return config

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
                choices=sorted(Task.values()),
            )
            # print(f"Task: {self.task}")
        if self._index == 1:
            # Mode
            self.mode = OptionPrompt.ask(
                prompt=ARGUMENTS.mode.prompt_text,
                default=self.mode,
                choices=sorted(RunMode.values()),
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
                default=DEVICE_MANAGER.get(self.device).name,
                choices=DEVICE_MANAGER.names,
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
    ConfigManager(root=Path.cwd(), prompt=True)


# endregion
