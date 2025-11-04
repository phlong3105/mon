#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements runtime utilities functions."""

_all__ = [
    "list_archs",
    "list_config_files",
    "list_datasets",
    "list_models",
    "list_weights_files",
    "load_config",
    "load_project_defaults",
    "parse_config_file",
    "parse_model_dir",
    "parse_model_fullname",
    "parse_output_dir",
    "parse_save_dir",
    "parse_weights_dir",
    "parse_weights_file",
    "parse_weights_from_config",
    "print_run_summary",
]

import importlib.util
import os
from typing import Any, Sequence

import box
import yaml

from mon.constants import DATASETS, MODELS, ROOT_DIR, ZOO_DIR
from mon.core.console import console, log, log_error, pprint_dict
from mon.core.enum import MLType, Split, Task
from mon.core.pathlib import Path
from mon.core.utils import depascalize, to_list, unique


# ----- Retrieval -----
def list_config_files(
    project_root : Path,
    model_root   : Path = None,
    model        : str  = None,
    absolute_path: bool = False,
) -> list[Path]:
    """Lists configuration files in the project and/or model directory.

    Args:
        project_root: Root directory of the project.
        model_root: Root directory of the model. Default: ``None``.
        model: Name of the model to filter configs. Default: ``None``.
        absolute_path: If ``True``, returns absolute paths else file names.
            Default: ``False``.

    Returns:
        Sorted list of config file paths.
    """
    def is_valid(x) -> bool:
        return x not in [None, "", "None"]

    def collect_config_files(root: Path | str) -> list[Path]:
        config_dir = Path(root) / "config"
        return [
            c for c in list(config_dir.files(recursive=True))
            if (f"{os.sep}archive{os.sep}"  not in str(c)) and
               (f"{os.sep}excluded{os.sep}" not in str(c))
        ]
    
    # List config files in project and model directories
    config_files = []
    if is_valid(project_root):
        config_files += collect_config_files(project_root)
    if is_valid(model_root):
        config_files += collect_config_files(model_root)
    
    # Filter
    config_files = [
        cf for cf in config_files
        if cf.is_config_file() or (cf.is_py_file() and cf.name != "__init__.py")
    ]
    
    if is_valid(model):
        config_files = [cf for cf in config_files if model in cf.name]
    
    if not absolute_path:
        config_files = [cf.name for cf in config_files]
      
    return sorted(unique(config_files))


def list_tasks(project_root: Path = None) -> list[str]:
    """Lists running tasks in the project.

    Args:
        project_root: Root directory of the project. Default: ``None`` means
            list all tasks supported in ``mon`` frameworks.

    Returns:
        Sorted list of task names as strings.
    """
    tasks = Task.names()
    
    if project_root:
        default_configs = load_project_defaults(project_root)
        default_tasks = default_configs.get("TASKS", [])
        if default_tasks not in [None, []]:
            tasks = default_tasks
    
    return sorted([t.value for t in tasks])


def list_archs(task: str = None, mode: str = None, project_root: Path = None) -> list[str]:
    """Lists all running archs in the project for a given task and mode.

    Args:
        task: Task to filter archs. Default: ``None``.
        mode: Mode of archs (``train`` or ``None``). Default: ``None``.
        project_root: Root directory of project. . Default: ``None`` means
            list all archs supported in ``mon`` frameworks.

    Returns:
        Sorted list of unique arch names matching task and mode.
    """
    models = list_models(task=task, mode=mode, project_root=project_root)
    
    default_configs = load_project_defaults(project_root=project_root)
    if default_configs.get("MODELS"):
        default_models = [depascalize(m) for m in default_configs["MODELS"]]
        models = [m for m in models if depascalize(m) in default_models]
    
    flatten_mon_models = MODELS.flatten_dict
    archs = [flatten_mon_models[m].arch for m in models]
    archs = [a.strip() for a in archs if a not in [None, "None", ""]]
    
    return sorted(unique(archs))


def list_models(task: str = None, mode: str = None,  arch: str = None, project_root: Path = None) -> list[str]:
    """Lists all running models in the project for a given task, mode, and arch.

    Args:
        task: Task to filter models. Default: ``None``.
        mode: Mode of models (``train`` or ``None``). Default: ``None``.
        arch: Arch to filter models. Default: ``None``.
        project_root: Root directory of project. . Default: ``None`` means
            list all models supported in ``mon`` frameworks.

    Returns:
        Sorted list of model names matching task, mode, and arch.
    """
    flatten_models = MODELS.flatten_dict
    models         = list(flatten_models.keys())
    
    if task in Task.values():
        task = Task(task)
        models = [m for m in models if task in flatten_models[m].tasks]
    if mode == "train":
        models = [m for m in models if any(lt in MLType.trainable() for lt in flatten_models[m].mltypes)]
    if arch:
        models = [m for m in models if arch == flatten_models[m].arch]
    
    default_configs = load_project_defaults(project_root=project_root)
    if default_configs.get("MODELS"):
        project_models = [depascalize(m) for m in default_configs["MODELS"]]
        models         = [m for m in models if depascalize(m) in project_models]
    
    return sorted(models)


def list_weights_files(model: str, project_root: Path = None) -> list[Path]:
    """Lists weights files for a model in project and ``zoo`` dirs.

    Args:
        model: Name of model to filter weights files.
        project_root: Root directory of project. . Default: ``None``.

    Returns:
        Sorted list of weights file paths.
    """
    def collect_weights_files(root: Path) -> list[Path]:
        return sorted(f for f in root.rglob("*") if f.is_weights_file())
    
    # List all weights files in the project root and ``zoo`` directories.
    weights_files: list[Path] = []
    if project_root not in [None, "None", ""]:
        weights_files += collect_weights_files(Path(project_root) / "run" / "train")
    weights_files += collect_weights_files(ZOO_DIR)
    
    # Filter weights files by model name.
    weights_files = [f for f in weights_files if model in f.parts]
    
    return sorted(unique(weights_files))


def list_datasets(task: str, mode: str, project_root: Path = None) -> list[str]:
    """Lists all datasets in the project for a given task and mode.

    Args:
        task: Task for which datasets are listed.
        mode: Mode of datasets (``train`` or ``test``).
        project_root: Root directory of project. . Default: ``None`` means
            list all datasets supported in ``mon`` frameworks.

    Returns:
        Sorted list of dataset names matching task and mode.
    """
    split    = Split("train" if mode == "train" else "test")
    task     = Task(task)
    datasets = sorted([
        d for d in DATASETS
        if task in DATASETS[d].tasks and split in DATASETS[d].splits
    ])
    
    default_configs = load_project_defaults(project_root)
    if default_configs.get("DATASETS"):
        datasets = [d for d in datasets if d in default_configs["DATASETS"]]
    return datasets


# ----- Read -----
def load_config(config: Any, verbose: bool = True) -> dict | box.Box:
    """Loads configuration from a file.

    Args:
        config: Config source (dict, file path, or string).
        verbose: Verbosity mode. Default: ``True``.

    Returns:
        Dict with loaded config, or empty dict if loading fails.
    """
    data = None
    if isinstance(config, box.Box):
        data = config
    elif isinstance(config, dict):
        data = box.Box(config)
    elif isinstance(config, Path | str):
        config = Path(config)
        if config.is_py_file(exist=True):
            spec   = importlib.util.spec_from_file_location(str(config.stem), str(config))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data   = {key: value for key, value in module.__dict__.items() if not key.startswith("__")}
        elif config.is_yaml_file(exist=True):
            with open(str(config), "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
    
    if verbose:
        if data:
            log(f"Loaded configuration from: {config}.")
        else:
            log_error(f"Could not load configuration from: {config}. Returning empty dict.")
            
    data = data or {}
    return box.Box(data)


def load_project_defaults(project_root: Path) -> dict:
    """Gets the default configuration of the project.

    Args:
        project_root: Root directory of the project.

    Returns:
        Dict with default config, or empty dict if invalid or not found.
    """
    if project_root in [None, "None", ""]:
        log_error(f"``project_root`` is not a valid project directory: {project_root}.")
        return {}
    
    config_file = Path(project_root) / "config" / "default.py"
    if not config_file.exists():
        return {}
    
    spec   = importlib.util.spec_from_file_location("default", str(config_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return {
        key: value
        for key, value in module.__dict__.items()
        if not key.startswith('__')
    }


# ----- Parsing -----
def parse_config_file(config: Path, project_root: Path, model_root: Path = None) -> Path | None:
    """Parses a config file from given components.

    Args:
        config: Config file path or name.
        project_root: Root directory of the project.
        model_root: Root directory of the model. Default: ``None``.

    Returns:
        Config file path if found, else ``None``.
    """
    def find_config_in_dirs(config, dirs):
        for config_dir in dirs:
            config_ = (config_dir / config.name).config_file()
            if config_.is_config_file():
                return config_
        return None
    
    if config:
        config = Path(config)
        if config.is_config_file():
            return config
        config_ = config.config_file()
        if config_.is_config_file():
            return config_
        if project_root:
            config_dirs = ([Path(project_root) / "config"] +
                           (Path(project_root) / "config").subdirs(recursive=True))
            config_ = find_config_in_dirs(config, config_dirs)
            if config_:
                return config_
        if model_root:
            config_dirs = ([Path(model_root) / "config"] +
                           (Path(model_root) / "config").subdirs(recursive=True))
            config_ = find_config_in_dirs(config, config_dirs)
            if config_:
                return config_
    
    log_error(
        f"Could not find configuration file given: "
        f"config={config}, project_root={project_root}, model_root={model_root}"
    )
    return None


def parse_model_dir(arch: str, model: str) -> Path | None:
    """Parses the model's directory from given components.

    Args:
        arch: Architecture of the model.
        model: Name of the model.

    Returns:
        Model directory path if found, else ``None``.
    """
    model_dir = MODELS[arch][model].model_dir
    return Path(model_dir) if model_dir else None


def parse_model_fullname(name: str, data: str, suffix: str = None) -> str:
    """Parses the model's full name as ``name-data-suffix`` from components.

    Args:
        name: Model's base name.
        data: BaseDataset name.
        suffix: Optional suffix for model name. Default: ``None``.

    Returns:
        Parsed full model name as a string.
    """
    if not name:
        log_error("[name] must be provided for the model.")
    
    fullname = name
    if data:
        fullname = f"{fullname}_{data}"
    if suffix:
        suffix_  = depascalize(suffix)
        if suffix_ not in fullname:
            fullname = f"{fullname}_{suffix_}"
    return fullname


def parse_weights_dir(root: Path, weights: Path | Sequence[Path]) -> Path | Sequence[Path]:
    """Parses directory containing weights from given components.
    
    Args:
        root: Root directory (e.g., project root).
        weights: Weights file(s) to parse (path or sequence of paths).
    
    Returns:
        Parsed weights path(s) as a single path or a sequence of paths, or ``None`` if empty.
    """
    root    = Path(root)
    weights = to_list(weights)
    
    for i, w in enumerate(weights):
        if w is not None:
            if (ROOT_DIR / w).is_dir():
                weights[i] = ROOT_DIR / w
            elif (root / w).is_dir():
                weights[i] = root / w
    
    weights = [Path(w) for w in weights if w not in [None, "None", ""]]
    
    if len(weights) == 1:
        return weights[0]
    return weights or None


def parse_weights_file(root: Path, weights: Path | Sequence[Path]) -> Path | Sequence[Path]:
    """Parses weights file path(s) from given components.
    
    Args:
        root: Root directory (e.g., project root).
        weights: Weights file(s) to parse (path or sequence of paths).

    Returns:
        Parsed weights path(s) as a single path or a sequence of paths, or ``None`` if empty.
    """
    root    = Path(root)
    weights = to_list(weights)
    
    for i, w in enumerate(weights):
        if w is not None:
            if (ROOT_DIR / w).is_weights_file():
                weights[i] = ROOT_DIR / w
            elif (root / w).is_weights_file():
                weights[i] = root / w
    
    weights = [Path(w) for w in weights if w not in [None, "None", ""]]
    
    if len(weights) == 1:
        return weights[0]
    return weights or None


def parse_weights_from_config(config: Path | dict) -> Path | None:
    """Gets the weights file path from a config file.

    Args:
        config: Config file path or a dictionary containing weights info.

    Returns:
        Weights file path or ``None`` if not found or invalid.
    """
    if config is None:
        return None
    
    if not Path(config).is_config_file(exist=True):
        return None
    
    args = load_config(config, False)
    weights = args.get("weights", None)
    return Path(weights) if weights else None


def parse_save_dir(
    root : Path,
    arch : str = None,
    model: str = None,
    data : str = None,
) -> Path:
    """Parses a save dir in format: root/arch/model/data.

    Args:
        root: Project root.
        arch: Model architecture. Default: ``None``.
        model: Model name. Default: ``None``.
        data: BaseDataset name. Default: ``None``.

    Returns:
        Parsed ``save_dir`` path.
    """
    save_dir = Path(root)
    data     = Path(data) if data not in [None, "None", ""] else None
    if arch:
        save_dir /= arch
    if model:
        save_dir /= model
        if isinstance(data, Path):
            if data.is_dir() or data.is_file():
                save_dir /= data.stem
            else:
                save_dir /= data
    return save_dir


def parse_output_dir(
    root        : Path,
    dirname     : Path | str,
    subdir_name : Path | str,
    src_path    : Path | str,
    keep_subdirs: bool = False,
    save_nearby : bool = False,
) -> Path:
    """Parses the output directory path from given components.

    It should be in this pattern:
        ``root/dirname/file``.
        where:
            root = ``parse_save_dir()``
    
    Args:
        root: Root directory. Root is assumed to be in this pattern: ``root/arch/model/[fullname or dirname]``.
        src_path: Source file path.
        dirname: Directory name.
        subdir_name: Subdirectory name. Default: ``None``.
        keep_subdirs: If ``True``, keeps subdirectories in the path. Default: ``False``.
        save_nearby: If ``True``, saves in the same parent directory as the file. Default: ``False``.

    Example:
        root     = ../enhance/run/predict/zerodce/zerodce/dicm
        dirname  = dicm
        src_path = ../enhance/data/dicm/test/image/0001.jpg
        rel_path = dicm/test/image/0001.jpg
        return   : ../enhance/run/predict/zerodce/zerodce/dicm/test/image
    """
    root        = Path(root)
    dirname     = Path(dirname)
    subdir_name = subdir_name if subdir_name not in [None, "None", ""] else None
    subdir_name = None if save_nearby else subdir_name
    src_path    = Path(src_path)

    # Update root and dirname
    if save_nearby:
        if root.stem == dirname.stem:
            root_suffix = root.parent.stem
        else:
            root_suffix = root.stem
        root    = src_path.parent.parent / f"{src_path.parent.stem}_{root_suffix}"
        dirname = Path(src_path.parent.stem)

    if keep_subdirs:
        rel_path = src_path.relative_path(dirname)
        if subdir_name:
            return root / subdir_name / rel_path.parent
        else:
            return root / rel_path.parent
    else:
        if not save_nearby and dirname.stem != root.stem:
            root = root / dirname.stem
        if subdir_name:
            return root / subdir_name
        else:
            return root


# ----- Print -----
def print_run_summary(args: dict | box.Box, full: bool = False):
    """Prints a summary of the run configuration.

    Args:
        args: Configuration arguments.
        full: If ``True``, prints all details. Default: ``False``.
    """
    if full:
        pprint_dict(args.to_dict() if isinstance(args, box.Box) else args)
    else:
        console.rule(f"[bold red]{args.fullname}")
        log(f"Machine   : {args.hostname}")
        log(f"Task      : {args.task}")
        log(f"Mode      : {args.mode}")
        log(f"Model     : {args.fullname}")
        log(f"Data      : {args.data}")
        log(f"Save Dir  : {args.save_dir}")
        log(f"Config    : {args.config}")
