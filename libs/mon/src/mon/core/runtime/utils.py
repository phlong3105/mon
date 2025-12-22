#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration and project utilities for CLI runtime.

This module provides helpers for listing, resolving, and loading runtime
configuration, models, datasets, and output directories.
"""

_all__ = [
    "find_archs",
    "find_config_files",
    "find_datasets",
    "find_models",
    "find_weights_files",
    "load_config",
    "load_project_defaults",
    "parse_config_file",
    "parse_data_dir",
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

from mon.core.console import console, log, log_error, pprint_dict
from mon.core.constants import ROOT_DIR, ZOO_DIR
from mon.core.enum import MLType, Split, Task
from mon.core.factory import DATASETS, MODELS
from mon.core.pathlib import Path
from mon.core.utils import depascalize, to_list, unique


# ==============================================================================
# DISCOVERY & REGISTRY LOOKUP
# ==============================================================================

# --- Resource Finders (Listing architectures, models, and tasks) ---
def list_archs(
    task        : str  = None,
    mode        : str  = None,
    project_root: Path = None
) -> list[str]:
    """Return available architectures for a task and mode.

    Filter available models by task, mode and project defaults to produce
    a list of architecture names.

    Args:
        task: Task name to filter architectures.
        mode: Run mode to filter architectures.
        project_root: Optional project root to apply project defaults.

    Returns:
        A sorted list of architecture names.
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


def list_models(
    task        : str  = None,
    mode        : str  = None,
    arch        : str  = None,
    project_root: Path = None
) -> list[str]:
    """Return available models filtered by task, mode and arch.

    Use the registered model registry and apply optional filters and
    project defaults.

    Args:
        task: Task name to filter models.
        mode: Run mode to filter models.
        arch: Architecture name to filter models.
        project_root: Optional project root to apply project defaults.

    Returns:
        A sorted list of model names.
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


def list_tasks(project_root: Path = None) -> list[str]:
    """Return available task names for the project.

    Consult project defaults if a project root is provided.

    Args:
        project_root: Optional project root to consult project defaults.

    Returns:
        A sorted list of task names.
    """
    tasks = Task.names()
    
    if project_root:
        default_configs = load_project_defaults(project_root)
        default_tasks = default_configs.get("TASKS", [])
        if default_tasks not in [None, []]:
            tasks = default_tasks
    
    return sorted([t.value for t in tasks])


# --- Weight & Data Locators (Searching project runs and the global zoo) ---
def list_config_files(
    project_root : Path,
    model_root   : Path = None,
    model        : str  = None,
    absolute_path: bool = False,
) -> list[Path]:
    """Return configuration files for project and model.

    List configuration files found under project and model config
    directories and return either filenames or absolute paths.

    Args:
        project_root: Project root path.
        model_root: Optional model-specific root to include.
        model: Optional model name to filter results.
        absolute_path: If True, return absolute Paths; otherwise return
            names.

    Returns:
        A sorted list of configuration file paths or filenames.
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


def list_datasets(task: str, mode: str, project_root: Path = None) -> list[str]:
    """Return dataset names for a task and mode.

    Determine datasets that support the task and required split, and
    apply project defaults if available.

    Args:
        task: Task name.
        mode: Run mode, e.g., "train" or "predict".
        project_root: Optional project root to apply defaults.

    Returns:
        A list of dataset names supporting the task and split.
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


def list_weights_files(model: str, project_root: Path = None) -> list[Path]:
    """Return available weight files for a model.

    Search the project runs and the global zoo directory and return
    matching weight files.

    Args:
        model: Model name to filter weights.
        project_root: Optional project root to include run/train outputs.

    Returns:
        A sorted list of matching weight file paths.
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


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

# --- Serialization Ops ---
def load_config(config: Any, verbose: bool = True) -> dict | box.Box:
    """Load configuration from a path, module, or mapping.

    Accept a mapping, a Box, or a filesystem path pointing to a Python or
    YAML configuration and return a loaded configuration mapping.

    Args:
        config: Mapping or path to a config file.
        verbose: If True, log load success or failure.

    Returns:
        The loaded configuration as a Box. Returns an empty Box if nothing
        is found.
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
    """Load a project's default configuration.

    Read the project's config/default.py and return the defined defaults.

    Args:
        project_root: Project root path.

    Returns:
        The defaults mapping loaded from the project's default.py, or an
        empty mapping if none is present.
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


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

# --- Directory Parsers ---
def parse_data_dir(root: Path, data_dir: Path = "") -> Path:
    """Resolve an absolute data directory path from candidates.

    Try a series of candidate locations and return the first existing
    directory. Raise an error if no candidate exists.

    Args:
        root: Project root.
        data_dir: Candidate data directory name or path.

    Returns:
        The first candidate directory path that exists.

    Raises:
        FileNotFoundError: If no candidate directory exists.
    """
    root_      = Path(root)     if root     not in [None, "None", ""] else ROOT_DIR
    data_dir_  = Path(data_dir) if data_dir not in [None, "None", ""] else None

    candidates = []
    if data_dir_:
        candidates.extend([
            data_dir_,
            root_    / data_dir_,
            root_    / "data" / data_dir_,
            ROOT_DIR / data_dir_,
            ROOT_DIR / "data" / data_dir_
        ])
    candidates.extend([
        root_    / "data",
        ROOT_DIR / "data"
    ])

    for d in candidates:
        if d.is_dir():
            return d
    raise FileNotFoundError(f"``data_dir`` not found: {data_dir}.")


def parse_model_dir(arch: str, model: str) -> Path | None:
    """Return the model directory for the given arch and model.

    Args:
        arch: Architecture name.
        model: Model name.

    Returns:
        The path to the model directory, or None if unspecified.
    """
    model_dir = MODELS[arch][model].model_dir
    return Path(model_dir) if model_dir else None


def parse_model_fullname(name: str, data: str, suffix: str = None) -> str:
    """Compose a model fullname from name, data, and optional suffix.

    Build a normalized fullname string by appending dataset and an
    optional suffix if not already present.

    Args:
        name: Base model name.
        data: Dataset or data identifier to append.
        suffix: Optional suffix to append.

    Returns:
        The composed fullname string.
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


def parse_save_dir(
    root : Path,
    arch : str = None,
    model: str = None,
    data : str = None,
) -> Path:
    """Build a save directory path from components.

    Combine root, architecture, model and optional data to construct a
    save directory path suitable for storing run outputs.

    Args:
        root: Base root path.
        arch: Optional architecture name.
        model: Optional model name.
        data: Optional data name or path.

    Returns:
        The constructed save directory path.
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
    """Compute the output directory for a source path.

    Determine where to place outputs for a given source path, optionally
    preserving subdirectory structure or saving outputs near the source.

    Args:
        root: Base save root.
        dirname: Directory name used in save structure.
        subdir_name: Optional subdirectory under root to place outputs.
        src_path: Source file path used to preserve subdir structure.
        keep_subdirs: If True, preserve subdirectories from src_path.
        save_nearby: If True, save outputs near the source path instead.

    Returns:
        The resolved output directory path.
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


def parse_weights_dir(root: Path, weights: Path | Sequence[Path]) -> Path | Sequence[Path]:
    """Resolve weight directories from root and weight names.

    Convert relative weight names to absolute directories under the
    project root or the global ROOT_DIR.

    Args:
        root: Project root path.
        weights: Weight name or iterable of weight names.

    Returns:
        A resolved path, a list of resolved paths, or None if nothing was
        found.
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


# --- Artifact Resolvers ---
def parse_config_file(config: Path, project_root: Path, model_root: Path = None) -> Path | None:
    """Resolve a config file path from given components.

    Search project and model config directories and return the first
    matching config file if found.

    Args:
        config: Candidate config name or path.
        project_root: Project root to search under.
        model_root: Optional model root to search under.

    Returns:
        The resolved config path if found, otherwise None.
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


def parse_weights_file(root: Path, weights: Path | Sequence[Path]) -> Path | Sequence[Path]:
    """Resolve weight file paths given root and weight names.

    Convert weight names to existing weight files under the project root
    or the global ROOT_DIR.

    Args:
        root: Project root path.
        weights: Weight file name or iterable of names.

    Returns:
        A resolved path, a list of resolved paths, or None if nothing was
        found.
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
    """Extract a weights path from a config file or mapping.

    Inspect the provided config and return the configured weights path
    if present.

    Args:
        config: Path to config or dict-like config.

    Returns:
        The weights path if present, otherwise None.
    """
    if config is None:
        return None
    
    if not Path(config).is_config_file(exist=True):
        return None
    
    args = load_config(config, False)
    weights = args.get("weights", None)
    return Path(weights) if weights else None


# ==============================================================================
# UTILS
# ==============================================================================

# --- Print ---
def print_run_summary(args: dict | box.Box, full: bool = False):
    """Print a concise summary of run arguments.

    Print a compact run summary or the full configuration when requested.

    Args:
        args: Arguments mapping (Box or dict).
        full: If True, pretty-print the full args and config.
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
