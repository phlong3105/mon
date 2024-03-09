#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

__all__ = [
    "list_config_files",
    "list_configs",
    "list_datasets",
    "list_extra_datasets",
    "list_extra_models",
    "list_models",
    "list_mon_datasets",
    "list_mon_models",
    "get_project_default_config",
    "list_tasks",
    "list_weights_files",
    "parse_menu_string",
]

import importlib.util
from typing import Collection, Sequence

import mon


# region Projects

def get_project_default_config(project_root: str | mon.Path) -> dict:
    if project_root in [None, "None", ""]:
        mon.error_console.log(f"{project_root} is not a valid project directory.")
        return {}
    
    config_file = mon.Path(project_root) / "config" / "default.py"
    if config_file.exists():
        spec   = importlib.util.spec_from_file_location("default", str(config_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {key: value for key, value in module.__dict__.items() if not key.startswith('__')}
    return {}

# endregion


# region Tasks

def list_tasks(project_root: str | mon.Path) -> list[str]:
    tasks           = mon.Task.keys()
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("TASKS", False) and len(default_configs["TASKS"]) > 0:
        tasks = [t for t in tasks if t in default_configs["TASKS"]]
    tasks = [t.value for t in tasks]
    return tasks
    
# endregion


# region Models

def list_mon_models(task: str) -> list[str]:
    task   = mon.Task(task)
    models = mon.MODELS
    return sorted([m for m in models if task in models[m]._tasks])


def list_extra_models(task: str) -> list[str]:
    task 	     = mon.Task(task)
    models_extra = mon.MODELS_EXTRA
    return sorted([m for m in models_extra if task in models_extra[m]["tasks"]])


def list_models(
    task        : str,
    project_root: str | mon.Path | None = None
) -> list[str]:
    models          = sorted(list_mon_models(task) + list_extra_models(task))
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("MODELS", False) and len(default_configs["MODELS"]) > 0:
        project_models = [mon.snakecase(m) for m in default_configs["MODELS"]]
        if len(project_models) > 0:
            models = [m for m in models if mon.snakecase(m) in project_models]
    return models

# endregion


# region Config

def list_config_files(
    project_root: str | mon.Path,
    model       : str | None = None
) -> list[mon.Path]:
    """List configuration files in the given :param:`project`."""
    assert project_root not in [None, "None", ""]
    project_root = mon.Path(project_root)
    config_dir   = project_root / "config"
    config_files = list(config_dir.files())
    config_files = [
        cf for cf in config_files
        if (
            cf.is_config_file() or
            cf.is_py_file() and cf.name != "__init__.py"
        )
    ]
    if model not in [None, "None", ""]:
        model        = f"{model}_"
        config_files = [cf for cf in config_files if model in cf.name]
    config_files = sorted(config_files)
    return config_files


def list_configs(
    project_root: str | mon.Path,
    model       : str | None = None
) -> list[str]:
    config_files = list_config_files(project_root=project_root, model=model)
    return [f.name for f in config_files]
    
# endregion


# region Datasets

def list_mon_datasets(task: str, mode: str) -> list[str]:
    if mode in ["train"]:
        split = mon.Split("train")
    else:
        split = mon.Split("test")
    task	 = mon.Task(task)
    datasets = mon.DATASETS
    return sorted([
        d for d in datasets
        if (task in datasets[d].tasks and split in datasets[d].splits)
    ])


def list_extra_datasets(task: str, mode: str) -> list[str]:
    if mode in ["train"]:
        split = mon.Split("train")
    else:
        split = mon.Split("test")
    task 	 = mon.Task(task)
    datasets = mon.DATASETS_EXTRA
    return sorted([
        d for d in datasets
        if (task in datasets[d]["tasks"] and split in datasets[d]["splits"])
    ])


def list_datasets(
    task        : str,
    mode        : str,
    project_root: str | mon.Path | None = None
) -> list[str]:
    datasets        = sorted(list_mon_datasets(task, mode) + list_extra_datasets(task, mode))
    default_configs = get_project_default_config(project_root=project_root)
    if default_configs.get("DATASETS", False) and len(default_configs["DATASETS"]) > 0:
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets
    
# endregion


# region Weights

def list_weights_files(
    model       : str,
    config      : str            | None = None,
    project_root: str | mon.Path | None = None,
) -> list[mon.Path]:
    files = []
    #
    if project_root not in [None, "None", ""]:
        project_root = mon.Path(project_root)
        train_dir    = project_root / "run" / "train"
        files        = sorted(list(train_dir.rglob(f"*{model}*/*")))
        files        = [f for f in files if f.is_weights_file()]
        if config not in [None, "None", ""]:
            config = mon.Path(config).stem
            files  = [f for f in files if config in str(f)]
    #
    for path in sorted(list(mon.ZOO_DIR.rglob(f"*{model}*"))):
        if path.is_weights_file():
            files.append(path)
    #
    files = mon.unique(files)
    files = sorted(files)
    return files

# endregion


# region Save Dir

def list_train_save_dirs(project_root: str | mon.Path) -> list[mon.Path]:
    project_root = mon.Path(project_root)
    train_dir    = project_root / "run" / "train"
    save_dirs    = sorted(list(train_dir.dirs()))
    return save_dirs

# endregion


# region Parsing

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
