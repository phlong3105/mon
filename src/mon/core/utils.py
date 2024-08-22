#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core Utilities Package.

This module implements various useful utilities functions and data structures.
"""

from __future__ import annotations

__all__ = [
    "Timer",
    "check_installed_package",
    "get_gpu_device_memory",
    "get_machine_memory",
    "get_project_default_config",
    "is_extra_model",
    "is_rank_zero",
    "list_archs",
    "list_config_files",
    "list_configs",
    "list_cuda_devices",
    "list_datasets",
    "list_devices",
    "list_extra_archs",
    "list_extra_datasets",
    "list_extra_models",
    "list_models",
    "list_mon_archs",
    "list_mon_datasets",
    "list_mon_models",
    "list_tasks",
    "list_weights_files",
    "load_config",
    "parse_config_file",
    "parse_device",
    "parse_menu_string",
    "parse_model_name",
    "parse_save_dir",
    "parse_weights_file",
    "pynvml_available",
    "set_device",
    "set_random_seed",
]

import importlib
import importlib.util
import os
import random
import time
from typing import Any, Collection, Sequence

import numpy as np
import psutil
import torch

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

from mon.globals import MemoryUnit
from mon.core import pathlib, dtype, file, humps


# region Config

def get_project_default_config(project_root: str | pathlib.Path) -> dict:
    if project_root in [None, "None", ""]:
        from mon.core.rich import error_console
        error_console.log(f"{project_root} is not a valid project directory.")
        return {}
    
    config_file = pathlib.Path(project_root) / "config" / "default.py"
    if config_file.exists():
        spec   = importlib.util.spec_from_file_location(
            "default", str(config_file)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            key: value
            for key, value in module.__dict__.items()
            if not key.startswith('__')
        }
    return {}


def list_config_files(
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    model       : str                = None
) -> list[pathlib.Path]:
    """List configuration files in the given :obj:`project`."""
    config_files = []
    if project_root not in [None, "None", ""]:
        project_root        = pathlib.Path(project_root)
        project_config_dir  = project_root / "config"
        config_files       += list(project_config_dir.files(recursive=True))
    if model_root not in [None, "None", ""]:
        model_root          = pathlib.Path(model_root)
        model_config_dir    = model_root / "config"
        config_files       += list(model_config_dir.files(recursive=True))
    #
    config_files = [
        cf for cf in config_files
        if (
            cf.is_config_file() or
            cf.is_py_file() and cf.name != "__init__.py"
        )
    ]
    if model not in [None, "None", ""]:
        model_name   = parse_model_name(model)
        config_files = [cf for cf in config_files if f"{model_name}" in cf.name]
    config_files = dtype.unique(config_files)
    config_files = sorted(config_files)
    return config_files


def list_configs(
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    model       : str                = None
) -> list[str]:
    config_files = list_config_files(
        project_root = project_root,
        model_root   = model_root,
        model        = model
    )
    config_files = [str(f.name) for f in config_files]
    config_files = dtype.unique(config_files)
    config_files = sorted(config_files, key=lambda x: (os.path.splitext(x)[1], x))
    return config_files


def parse_config_file(
    config      : str | pathlib.Path,
    project_root: str | pathlib.Path,
    model_root  : str | pathlib.Path = None,
    weights_path: str | pathlib.Path = None,
) -> pathlib.Path | None:
    # assert config not in [None, "None", ""]
    if config not in [None, "None", ""]:
        # Check ``config`` itself
        config = pathlib.Path(config)
        if config.is_config_file():
            return config
        # Check for other config file extensions in the same directory
        config_ = config.config_file()
        if config_.is_config_file():
            return config_
        # Check for config file in ``'config'`` directory in ``project_root``.
        if project_root not in [None, "None", ""]:
            config_dirs  = [pathlib.Path(project_root / "config")]
            config_dirs += pathlib.Path(project_root / "config").subdirs(recursive=True)
            for config_dir in config_dirs:
                config_ = (config_dir / config.name).config_file()
                if config_.is_config_file():
                    return config_
        # Check for config file in ``'config'`` directory in ``model_root``.
        if model_root not in [None, "None", ""]:
            config_dirs  = [pathlib.Path(model_root / "config")]
            config_dirs += pathlib.Path(model_root / "config").subdirs(recursive=True)
            for config_dir in config_dirs:
                config_ = (config_dir / config.name).config_file()
                if config_.is_config_file():
                    return config_
    # Check for config file that comes along with ``weights_path``.
    if weights_path not in [None, "None", ""]:
        weights_path = weights_path[0] if isinstance(weights_path, list) else weights_path
        weights_path = pathlib.Path(weights_path)
        if weights_path.is_weights_file():
            config_ = (weights_path.parent / "config.py").config_file()
            if config_.is_config_file():
                return config_
    # That's it.
    from mon.core.rich import error_console
    error_console.log(f"No configuration is found at {config}.")
    return None  # config


def load_config(config: Any) -> dict:
    if config is None:
        data = None
    elif isinstance(config, dict):
        data = config
    elif isinstance(config, pathlib.Path | str):
        config = pathlib.Path(config)
        if config.is_py_file():
            spec   = importlib.util.spec_from_file_location(str(config.stem), str(config))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data  = {
                key: value
                for key, value in module.__dict__.items()
                if not key.startswith("__")
            }
        else:
            data = file.read_from_file(path=config)
    else:
        data = None
    
    if data is None:
        from mon.core.rich import error_console
        error_console.log(
            f"No configuration is found at {config}. Setting an empty dictionary."
        )
        data = {}
    return data

# endregion


# region Datasets

def list_mon_datasets(task: str, mode: str) -> list[str]:
    from mon.globals import Task, Split, DATASETS
    if mode in ["train"]:
        split = Split("train")
    else:
        split = Split("test")
    task	 = Task(task)
    datasets = DATASETS
    return sorted([
        d for d in datasets
        if (task in datasets[d].tasks and split in datasets[d].splits)
    ])


def list_extra_datasets(task: str, mode: str) -> list[str]:
    from mon.globals import Task, Split, EXTRA_DATASETS
    if mode in ["train"]:
        split = Split("train")
    else:
        split = Split("test")
    task 	 = Task(task)
    datasets = EXTRA_DATASETS
    return sorted([
        d for d in datasets
        if (task in datasets[d]["tasks"] and split in datasets[d]["splits"])
    ])


def list_datasets(
    task        : str,
    mode        : str,
    project_root: str | pathlib.Path = None
) -> list[str]:
    datasets = sorted(
          list_mon_datasets(task, mode)
        + list_extra_datasets(task, mode)
    )
    default_configs = get_project_default_config(project_root=project_root)
    if (
        default_configs.get("DATASETS", False)
        and len(default_configs["DATASETS"]) > 0
    ):
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets

# endregion


# region Device

def is_rank_zero() -> bool:
    """From Pytorch Lightning Official Document on DDP, we know that PL
    intended call the main script multiple times to spin off the child
    processes that take charge of GPUs.

    They used the environment variable "LOCAL_RANK" and "NODE_RANK" to denote
    GPUs. So we can add conditions to bypass the code blocks that we don't want
    to get executed repeatedly.
    """
    return True if (
        "LOCAL_RANK" not in os.environ.keys() and
        "NODE_RANK"  not in os.environ.keys()
    ) else False


def list_cuda_devices() -> str | None:
    """List all available cuda devices in the current machine."""
    if torch.cuda.is_available():
        cuda_str    = "cuda:"
        num_devices = torch.cuda.device_count()
        # gpu_devices = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        for i in range(num_devices):
            cuda_str += f"{i},"
        if cuda_str[-1] == ",":
            cuda_str = cuda_str[:-1]
        return cuda_str
    return None


def list_devices() -> list[str]:
    """List all available devices in the current machine."""
    devices: list[str] = []
    
    # Get CPU device
    devices.append("auto")
    devices.append("cpu")
    
    # Get GPU devices if available
    if torch.cuda.is_available():
        # All GPU devices
        all_cuda_str = "cuda:"
        num_devices  = torch.cuda.device_count()
        # gpu_devices = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        for i in range(num_devices):
            all_cuda_str += f"{i},"
            devices.append(f"cuda:{i}")
        
        if all_cuda_str[-1] == ",":
            all_cuda_str = all_cuda_str[:-1]
        if all_cuda_str != "cuda:0":
            devices.append(all_cuda_str)
    
    return devices


def set_device(device: Any, use_single_device: bool = True) -> torch.device:
    """Set a cuda device in the current machine.
    
    Args:
        device: Cuda devices to set.
        use_single_device: If ``True``, set a single-device cuda device in the list.
    
    Returns:
        A cuda device in the current machine.
    """
    device = parse_device(device)
    device = device[0] if isinstance(device, list) and use_single_device else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    return device


def get_machine_memory(unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Return the RAM status as a :obj:`list` of `[total, used, free]`.
    
    Args:
        unit: The memory unit. Default: ``'GB'``.
    """
    memory = psutil.virtual_memory()
    unit   = MemoryUnit.from_value(value=unit)
    ratio  = MemoryUnit.byte_conversion_mapping()[unit]
    total  = memory.total     / ratio
    free   = memory.available / ratio
    used   = memory.used      / ratio
    return [total, used, free]


def get_gpu_device_memory(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Return the GPU memory status as a :obj:`list` of `[total, used, free]`.
    
    Args:
        device: The index of the GPU device. Default: ``0``.
        unit: The memory unit. Default: ``'GB'``.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit.from_value(value=unit)
    h     = pynvml.nvmlDeviceGetHandleByIndex(index=device)
    info  = pynvml.nvmlDeviceGetMemoryInfo(h)
    ratio = MemoryUnit.byte_conversion_mapping()[unit]
    total = info.total / ratio
    free  = info.free  / ratio
    used  = info.used  / ratio
    return [total, used, free]


def parse_device(device: Any) -> list[int] | int | str:
    if isinstance(device, torch.device):
        return device
    
    device = device or None
    if device in [None, "", "cpu"]:
        device = "cpu"
    elif device in ["mps", "mps:0"]:
        device = device
    elif isinstance(device, int):
        device = [device]
    elif isinstance(device, str):  # Not ["", "cpu"]
        device = device.lower()
        for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
            device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
        if "," in device:
            device = [int(x) for x in device.split(",")]
        device = [0] if len(device) == 0 else device
    return device
# endregion


# region Menu

def parse_menu_string(items: Sequence | Collection, num_columns: int = 4) -> str:
    s = f"\n  "
    for i, item in enumerate(items):
        s += f"{f'{i}.':>6} {item}\n  "
    s += f"{f'Other.':} (please specify)\n  "
    
    '''
    w, h = mon.get_terminal_size()
    w 	 = w if w >= 80 else 80
    items_per_row = w // (padding + 2)
    padding = math.floor(w / num_columns) - 8
    
    s   = f"\n  "
    row = f""
    for i, item in enumerate(items):
        if i > 0 and i % num_columns == 0:
            s   += f"{row}\n\t"
            row  = f""
        else:
            t    = f"{f'{i}.':>4}{item}"
            row += f"{t:<{padding}}"
    if row != "":
        s += f"{row}\n\t"
    '''
    
    return s

# endregion


# region Models

def is_extra_model(model: str) -> bool:
    from mon.globals import MODELS, EXTRA_MODELS, EXTRA_MODEL_STR
    use_extra_model = f"{EXTRA_MODEL_STR}" in model
    model           = model.replace(f" {EXTRA_MODEL_STR}", "").strip()
    mon_models      = dtype.flatten_models_dict(MODELS)
    extra_models    = dtype.flatten_models_dict(EXTRA_MODELS)
    return (
        use_extra_model or
        (model not in mon_models and model in extra_models)
    )


def list_mon_models(
    task: str = None,
    mode: str = None,
    arch: str = None,
) -> list[str]:
    from mon.globals import Task, MODELS, Scheme
    flatten_models = dtype.flatten_models_dict(MODELS)
    task   = Task(task)   if task not in [None, "None", ""] else None
    mode   = Scheme(mode) if mode in ["online", "instance"] else None
    arch   = arch         if arch not in [None, "None", ""] else None
    models = list(flatten_models.keys())
    if task:
        models = [m for m in models if task in flatten_models[m].tasks]
    if mode:
        models = [m for m in models if mode in flatten_models[m]._schemes]
    if arch:
        models = [m for m in models if arch in flatten_models[m].arch]
    return sorted(models)


def list_extra_models(
    task: str = None,
    mode: str = None,
    arch: str = None,
) -> list[str]:
    from mon.globals import Task, EXTRA_MODELS, Scheme
    flatten_models = dtype.flatten_models_dict(EXTRA_MODELS)
    task   = Task(task)   if task not in [None, "None", ""] else None
    mode   = Scheme(mode) if mode in ["online", "instance"] else None
    arch   = arch         if arch not in [None, "None", ""] else None
    models = list(flatten_models.keys())
    if task:
        models = [m for m in models if task in flatten_models[m]["tasks"]]
    if mode:
        models = [m for m in models if mode in flatten_models[m]["schemes"]]
    if arch:
        models = [m for m in models if arch in flatten_models[m]["arch"]]
    return sorted(models)


def list_models(
    task        : str = None,
    mode        : str = None,
    arch        : str = None,
    project_root: str | pathlib.Path = None
) -> list[str]:
    from mon.globals import EXTRA_MODEL_STR
    models          = list_mon_models(task, mode, arch)
    extra_models    = list_extra_models(task, mode, arch)
    default_configs = get_project_default_config(project_root=project_root)
    if (
        default_configs.get("MODELS", False)
        and len(default_configs["MODELS"]) > 0
    ):
        project_models = [humps.snakecase(m) for m in default_configs["MODELS"]]
        if len(project_models) > 0:
            models = [
                m for m in models
                if humps.snakecase(m) in project_models
            ]
            extra_models = [
                m for m in extra_models
                if humps.snakecase(m) in project_models
            ]
    # Rename extra models for clarity
    for i, m in enumerate(extra_models):
        if m in models:
            extra_models[i] = f"{m} {EXTRA_MODEL_STR}"
    models = models + extra_models
    return sorted(models)


def list_mon_archs(task: str = None, mode: str = None) -> list[str]:
    from mon.globals import Task, MODELS, Scheme
    flatten_models = dtype.flatten_models_dict(MODELS)
    task   = Task(task)   if task not in [None, "None", ""] else None
    mode   = Scheme(mode) if mode in ["online", "instance"] else None
    models = list(flatten_models.keys())
    if task:
        models = [m for m in models if task in flatten_models[m].tasks]
    if mode:
        models = [m for m in models if mode in flatten_models[m]._schemes]
    archs = [flatten_models[m].arch for m in models]
    archs = [a.strip() for a in archs]
    archs = [a for a in archs if a not in [None, "None", ""]]
    return sorted(dtype.unique(archs))


def list_extra_archs(task: str = None, mode: str = None) -> list[str]:
    from mon.globals import Task, EXTRA_MODELS, Scheme
    flatten_models = dtype.flatten_models_dict(EXTRA_MODELS)
    task   = Task(task)   if task not in [None, "None", ""] else None
    mode   = Scheme(mode) if mode in ["online", "instance"] else None
    models = list(flatten_models.keys())
    if task:
        models = [m for m in models if task in flatten_models[m]["tasks"]]
    if mode:
        models = [m for m in models if mode in flatten_models[m]["schemes"]]
    archs = [flatten_models[m]["arch"] for m in models]
    archs = [a.strip() for a in archs]
    archs = [a for a in archs if a not in [None, "None", ""]]
    return sorted(dtype.unique(archs))


def list_archs(
    task        : str = None,
    mode        : str = None,
    project_root: str | pathlib.Path = None
) -> list[str]:
    from mon.globals import MODELS, EXTRA_MODELS
    models          = list_mon_models(task, mode)
    extra_models    = list_extra_models(task, mode)
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("MODELS", False) and len(default_configs["MODELS"]) > 0:
        project_models = [humps.snakecase(m) for m in default_configs["MODELS"]]
        if len(project_models) > 0:
            models       = [
                m for m in models
                if humps.snakecase(m) in project_models
            ]
            extra_models = [
                m for m in extra_models
                if humps.snakecase(m) in project_models
            ]
    #
    flatten_mon_models   = dtype.flatten_models_dict(MODELS)
    flatten_extra_models = dtype.flatten_models_dict(EXTRA_MODELS)
    mon_archs   = [flatten_mon_models[m].arch for m in models]
    extra_archs = [flatten_extra_models[m]["arch"] for m in extra_models]
    archs       = mon_archs + extra_archs
    archs       = [a.strip() for a in archs]
    archs       = [a for a in archs if a not in [None, "None", ""]]
    return sorted(dtype.unique(archs))


def parse_model_name(model: str) -> str:
    from mon.globals import EXTRA_MODEL_STR
    return model.replace(f" {EXTRA_MODEL_STR}", "").strip()

# endregion


# region Package

def check_installed_package(package_name: str, verbose: bool = False) -> bool:
    try:
        importlib.import_module(package_name)
        if verbose:
            print(f"`{package_name}` is installed.")
        return True
    except ImportError:
        if verbose:
            print(f"`{package_name}` is not installed.")
        return False

# endregion


# region Save Dir

def list_train_save_dirs(root: str | pathlib.Path) -> list[pathlib.Path]:
    root      = pathlib.Path(root)
    train_dir = root / "run" / "train"
    save_dirs = sorted(list(train_dir.dirs()))
    return save_dirs


def parse_save_dir(
    root   : str | pathlib.Path,
    arch   : str = None,
    model  : str = None,
    data   : str = None,
    project: str = None,
    variant: str = None,
) -> str | pathlib.Path:
    """Parse save_dir in the following format:
    ```
        root              | root
         |_ arch          |  |_ arch
             |_ model     |      |_ project
                 |_ data  |          |_ variant
    ```
    
    Args:
        root: The project root.
        arch: The model's architecture.
        model: The model's name.
        data: The dataset's name.
        project: The project's name. Usually used to perform ablation studies.
            Default is ``None``.
        variant: The variant's name. Usually used to perform ablation studies.
            Default is ``None``.
    """
    save_dir = pathlib.Path(root)
    if arch not in [None, "None", ""]:
        save_dir /= arch
    if project not in [None, "None", ""]:
        save_dir = save_dir / project
        if variant not in [None, "None", ""]:
            save_dir = save_dir / variant
    elif model not in [None, "None", ""]:
        save_dir = save_dir / model
        if data not in [None, "None", ""]:
            save_dir = save_dir / data
    return save_dir

# endregion


# region Seed

def set_random_seed(seed: int | list[int] | tuple[int, int]):
    """Set random seeds."""
    if isinstance(seed, list | tuple):
        if len(seed) == 2:
            seed = random.randint(seed[0], seed[1])
        else:
            seed = seed[-1]
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# endregion


# region Tasks

def list_tasks(project_root: str | pathlib.Path) -> list[str]:
    from mon.globals import Task
    tasks           = Task.keys()
    default_configs = get_project_default_config(project_root=project_root)
    if (
        default_configs.get("TASKS", False)
        and len(default_configs["TASKS"]) > 0
    ):
        tasks = [t for t in tasks if t in default_configs["TASKS"]]
    tasks = [t.value for t in tasks]
    return tasks

# endregion


# region Timer

class Timer:
    """A simple timer.
    
    Attributes:
        start_time: The start time of the current call.
        end_time: The end time of the current call.
        total_time: The total time of the timer.
        calls: The number of calls.
        diff_time: The difference time of the call.
        avg_time: The total average time.
    """
    
    def __init__(self):
        self.start_time = 0.0
        self.end_time   = 0.0
        self.total_time = 0.0
        self.calls      = 0
        self.diff_time  = 0.0
        self.avg_time   = 0.0
        self.duration   = 0.0
    
    @property
    def total_time_m(self) -> float:
        return self.total_time / 60.0
    
    @property
    def total_time_h(self) -> float:
        return self.total_time / 3600.0
    
    @property
    def avg_time_m(self) -> float:
        return self.avg_time / 60.0
    
    @property
    def avg_time_h(self) -> float:
        return self.avg_time / 3600.0
    
    @property
    def duration_m(self) -> float:
        return self.duration / 60.0
    
    @property
    def duration_h(self) -> float:
        return self.duration / 3600.0
    
    def start(self):
        self.clear()
        self.tick()
    
    def end(self) -> float:
        self.tock()
        return self.avg_time
    
    def tick(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()
    
    def tock(self, average: bool = True) -> float:
        self.end_time    = time.time()
        self.diff_time   = self.end_time - self.start_time
        self.total_time += self.diff_time
        self.calls      += 1
        self.avg_time    = self.total_time / self.calls
        if average:
            self.duration = self.avg_time
        else:
            self.duration = self.diff_time
        return self.duration
    
    def clear(self):
        self.start_time = 0.0
        self.end_time   = 0.0
        self.total_time = 0.0
        self.calls      = 0
        self.diff_time  = 0.0
        self.avg_time   = 0.0
        self.duration   = 0.0

# endregion


# region Weights

def list_weights_files(
    model       : str,
    project_root: str | pathlib.Path = None,
) -> list[pathlib.Path]:
    from mon.globals import ZOO_DIR
    
    weights_files = []
    # Search for weights in project_root
    if project_root not in [None, "None", ""]:
        project_root  = pathlib.Path(project_root)
        train_dir     = project_root / "run" / "train"
        weights_files = sorted(list(train_dir.rglob(f"*")))
        weights_files = [f for f in weights_files if f.is_weights_file()]
    # Search for weights in ZOO_DIR
    zoo_dir = ZOO_DIR
    for path in sorted(list(zoo_dir.rglob(f"*"))):
        if path.is_weights_file():
            weights_files.append(path)
    # Remove duplicate and sort
    model_name    = parse_model_name(model)
    weights_files = [f for f in weights_files if f"{model_name}" in str(f)]
    weights_files = dtype.unique(weights_files)
    weights_files = sorted(weights_files)
    return weights_files


def parse_weights_file(
    weights: str | pathlib.Path | Sequence[str | pathlib.Path]
) -> str | pathlib.Path | Sequence[str | pathlib.Path]:
    """Parse weights file. If the weights file is a relative path in the ``zoo``
    directory, then it will be converted to the absolute path. If the weights
    file is a list with a single weights files, then it will be converted to a
    single weights.
    
    Args:
        weights: The weights file to parse.
    """
    from mon.globals import ZOO_DIR
    weights = dtype.to_list(weights)
    for i, w in enumerate(weights):
        w = pathlib.Path(w)
        if not w.is_weights_file():
            if w.parts[0] in ["zoo"]:
                weights[i] = ZOO_DIR.parent / w
            else:
                weights[i] = ZOO_DIR / w
    
    if isinstance(weights, list | tuple):
        weights = None       if len(weights) == 0 else weights
        weights = weights[0] if len(weights) == 1 else weights
    return weights

# endregion
