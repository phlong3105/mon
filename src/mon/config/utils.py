#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module offers configuration files handling capabilities."""

from __future__ import annotations

__all__ = [
    "get_project_default_config",
    "is_extra_model",
    "list_config_files",
    "list_configs",
    "list_datasets",
    "list_extra_datasets",
    "list_extra_models",
    "list_models",
    "list_mon_datasets",
    "list_mon_models",
    "list_tasks",
    "list_weights_files",
    "load_config",
    "parse_config_file",
    "parse_model_name",
]

import importlib.util
import os
from typing import Any

from mon import core

console       = core.console
error_console = core.error_console


# region Projects

def get_project_default_config(project_root: str | core.Path) -> dict:
    if project_root in [None, "None", ""]:
        error_console.log(f"{project_root} is not a valid project directory.")
        return {}
    
    config_file = core.Path(project_root) / "config" / "default.py"
    if config_file.exists():
        spec   = importlib.util.spec_from_file_location("default", str(config_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {key: value for key, value in module.__dict__.items() if not key.startswith('__')}
    return {}

# endregion


# region Tasks

def list_tasks(project_root: str | core.Path) -> list[str]:
    from mon.globals import Task
    tasks           = Task.keys()
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("TASKS", False) and len(default_configs["TASKS"]) > 0:
        tasks = [t for t in tasks if t in default_configs["TASKS"]]
    tasks = [t.value for t in tasks]
    return tasks

# endregion


# region Models

def is_extra_model(model: str) -> bool:
    from mon.globals import MODELS, EXTRA_MODELS, EXTRA_MODEL_STR
    use_extra_model = f"{EXTRA_MODEL_STR}" in model
    model           = model.replace(f" {EXTRA_MODEL_STR}", "")
    return (
        use_extra_model or
        (model in MODELS and model not in EXTRA_MODELS)
    )


def list_mon_models(task: str, mode: str) -> list[str]:
    from mon.globals import Task, MODELS, Scheme
    task   = Task(task)
    models = MODELS
    models = [m for m in models if task in models[m]._tasks]
    if mode in ["online", "instance"]:
        mode   = Scheme(mode)
        models = [m for m in models if mode in models[m]._schemes]
    return sorted(models)


def list_extra_models(task: str, mode: str) -> list[str]:
    from mon.globals import Task, EXTRA_MODELS, Scheme
    task   = Task(task)
    models = EXTRA_MODELS
    models = [m for m in models if task in models[m]["tasks"]]
    if mode in ["online", "instance"]:
        mode   = Scheme(mode)
        models = [m for m in models if mode in models[m]["schemes"]]
    return sorted(models)


def list_models(
    task        : str,
    mode        : str,
    project_root: str | core.Path | None = None
) -> list[str]:
    from mon.globals import EXTRA_MODEL_STR
    models          = list_mon_models(task, mode)
    extra_models    = list_extra_models(task, mode)
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("MODELS", False) and len(default_configs["MODELS"]) > 0:
        project_models = [core.snakecase(m) for m in default_configs["MODELS"]]
        if len(project_models) > 0:
            models       = [m for m in models       if core.snakecase(m) in project_models]
            extra_models = [m for m in extra_models if core.snakecase(m) in project_models]
    # Rename extra models for clarity
    for i, m in enumerate(extra_models):
        # if m in models:
        extra_models[i] = f"{m}_{EXTRA_MODEL_STR}"
    models = models + extra_models
    return sorted(models)


def parse_model_name(model: str) -> str:
    from mon.globals import EXTRA_MODEL_STR
    return model.replace(f"_{EXTRA_MODEL_STR}", "").strip()

# endregion


# region Config

def list_config_files(project_root: str | core.Path, model: str | None = None) -> list[core.Path]:
    """List configuration files in the given :param:`project`."""
    assert project_root not in [None, "None", ""]
    project_root = core.Path(project_root)
    config_dir   = project_root / "config"
    config_files = list(config_dir.files(recursive=True))
    config_files = [
        cf for cf in config_files
        if (
            cf.is_config_file() or
            cf.is_py_file() and cf.name != "__init__.py"
        )
    ]
    if model not in [None, "None", ""]:
        model_name   = parse_model_name(model)
        config_files = [cf for cf in config_files if f"{model_name}_" in cf.name]
    config_files = core.unique(config_files)
    config_files = sorted(config_files)
    return config_files


def list_configs(project_root: str | core.Path, model: str | None = None) -> list[str]:
    from mon.globals import EXTRA_MODEL_STR
    config_files = list_config_files(project_root=project_root, model=model)
    config_files = [str(f.name) for f in config_files]
    # if is_extra_model(model):
    #     config_files = [EXTRA_MODEL_STR in f for f in config_files]
    config_files = core.unique(config_files)
    config_files = sorted(config_files, key=lambda x: (os.path.splitext(x)[1], x))
    return config_files


def parse_config_file(config: str | core.Path, project_root: str | core.Path) -> core.Path | None:
    # assert config not in [None, "None", ""]
    if config in [None, "None", ""]:
        error_console.log(f"No configuration given.")
        return None
    #
    config = core.Path(config)
    if config.is_config_file():
        return config
    #
    config_ = config.config_file()
    if config_.is_config_file():
        return config_
    #
    config_dirs = core.Path(project_root).subdirs(recursive=True)
    for config_dir in config_dirs:
        config_ = config_dir / config.name
        if config_.is_config_file():
            return config_
    for config_dir in config_dirs:
        config_ = (config_dir / config.name).config_file()
        if config_.is_config_file():
            return config_
    #
    error_console.log(f"No configuration is found at {config}.")
    return None  # config


def load_config(config: Any) -> dict:
    if config is None:
        data = None
    elif isinstance(config, dict):
        data = config
    elif isinstance(config, core.Path | str):
        config = core.Path(config)
        if config.is_py_file():
            spec   = importlib.util.spec_from_file_location(str(config.stem), str(config))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data  = {key: value for key, value in module.__dict__.items() if not key.startswith("__")}
        else:
            data = core.read_from_file(path=config)
    else:
        data = None
    
    if data is None:
        error_console.log(f"No configuration is found at {config}. Setting an empty dictionary.")
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
    project_root: str | core.Path | None = None
) -> list[str]:
    datasets        = sorted(list_mon_datasets(task, mode) + list_extra_datasets(task, mode))
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("DATASETS", False) and len(default_configs["DATASETS"]) > 0:
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets
    
# endregion


# region Weights

def list_weights_files(model: str, project_root: str | core.Path | None = None) -> list[core.Path]:
    from mon.globals import ZOO_DIR
    
    weights_files = []
    # Search for weights in project_root
    if project_root not in [None, "None", ""]:
        project_root  = core.Path(project_root)
        train_dir     = project_root / "run" / "train"
        weights_files = sorted(list(train_dir.rglob(f"*")))
        weights_files = [f for f in weights_files if f.is_weights_file()]
    # Search for weights in ZOO_DIR
    zoo_dir = ZOO_DIR / "mon_extra" if is_extra_model(model) else ZOO_DIR / "mon"
    for path in sorted(list(zoo_dir.rglob(f"*"))):
        if path.is_weights_file():
            weights_files.append(path)
    # Remove duplicate and sort
    model_name    = parse_model_name(model)
    weights_files = [f for f in weights_files if f"{model_name}_" in str(f)]
    weights_files = core.unique(weights_files)
    weights_files = sorted(weights_files)
    return weights_files

# endregion


# region Save Dir

def list_train_save_dirs(project_root: str | core.Path) -> list[core.Path]:
    project_root = core.Path(project_root)
    train_dir    = project_root / "run" / "train"
    save_dirs    = sorted(list(train_dir.dirs()))
    return save_dirs

# endregion
