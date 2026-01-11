#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration and project utilities for CLI runtime.

This module provides helpers for listing, resolving, and loading runtime
configuration.
"""

from __future__ import annotations

__all__ = [
    "list_archs",
    "list_config_files",
    "list_datasets",
    "list_models",
    "list_tasks",
    "list_weights_files",
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
from typing import Any, Optional, Sequence

import box
import yaml

from mon.core.console import console, log, log_error, pprint_dict
from mon.core.constants import ROOT_DIR, ZOO_DIR
from mon.core.enum import MLType, Split, Task
from mon.core.factory import DATASETS, MODELS
from mon.core.pathlib import Path
from mon.core.utils import depascalize, to_list


# ==============================================================================
# region DISCOVERY
# ==============================================================================

def list_archs(
    task        : str | None  = None,
    mode        : str | None  = None,
    project_root: Path | None = None
) -> list[str]:
    """Return available architectures for a task and mode.

    Filter available models by task, mode and project defaults to produce
    a list of architecture names.

    Args:
        task: Task name to filter architectures. Defaults to None.
        mode: Run mode to filter architectures. Defaults to None.
        project_root: Optional project root to apply project defaults.
            Defaults to None.

    Returns:
        Sorted list of architecture names.
    """
    # Get base model list
    models = list_models(task=task, mode=mode, project_root=project_root)
    
    # Filter by project defaults (if they exist)
    default_configs = load_project_defaults(project_root=project_root)
    allowed_models  = default_configs.get("MODELS")
    
    if allowed_models:
        # Use a set for O(1) lookups
        default_set = {depascalize(m) for m in allowed_models}
        models      = [m for m in models if depascalize(m) in default_set]
    
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
    
    return sorted(list(archs))


def list_models(
    task        : str | None  = None,
    mode        : str | None  = None,
    arch        : str | None  = None,
    project_root: Path | None = None
) -> list[str]:
    """Return available models filtered by task, mode and arch.

    Use the registered model registry and apply optional filters and
    project defaults.

    Args:
        task: Task name to filter models. Defaults to None.
        mode: Run mode to filter models. Defaults to None.
        arch: Architecture name to filter models. Defaults to None.
        project_root: Optional project root to apply project defaults.
            Defaults to None.

    Returns:
        Sorted list of model names.
    """
    # Access the flat registry view
    flatten_models = MODELS.flatten_dict
    models         = list(flatten_models.keys())
    
    # Filter by Task (e.g., Segmentation, Classification)
    if task and task in Task.values():
        task_enum = Task(task)
        models    = [m for m in models if task_enum in flatten_models[m].tasks]
    
    # Filter by Mode (e.g., can this model be trained?)
    if mode == "train":
        trainable_types = set(MLType.trainable())
        models = [m for m in models if any(lt in trainable_types
                                           for lt in flatten_models[m].mltypes)]
    
    # Filter by Architecture (e.g., resnet50)
    if arch:
        models = [m for m in models if arch == flatten_models[m].arch]
    
    # Apply Project Scoping (Restrict models to specific project-approved versions)
    default_configs = load_project_defaults(project_root=project_root)
    project_allowed = default_configs.get("MODELS")
    if project_allowed:
        # Normalize for case-insensitive matching
        project_set = {depascalize(m) for m in project_allowed}
        models      = [m for m in models if depascalize(m) in project_set]
    
    return sorted(models)


def list_tasks(project_root: Path | None = None) -> list[str]:
    """Return available task names for the project.

    Consult project defaults if a project root is provided.

    Args:
        project_root: Optional project root to consult project defaults.
            Defaults to None.

    Returns:
        Sorted list of task names.
    """
    # Start with global defaults
    # Assuming Task.names() returns a list of Enum objects or strings
    tasks = Task.names()
    
    # Consult project-specific restrictions
    if project_root:
        default_configs = load_project_defaults(project_root)
        project_tasks   = default_configs.get("TASKS")
        
        # Ensure we have a non-empty list
        if project_tasks:
            tasks = project_tasks
    
    # Normalize to string values and remove duplicates
    # This handles both Enum objects (t.value) and raw strings
    output = set()
    for t in tasks:
        val = t.value if hasattr(t, "value") else str(t)
        output.add(val.lower().strip())
        
    return sorted(list(output))


def list_config_files(
    project_root : Path,
    model_root   : Path | None = None,
    model        : str | None  = None,
    absolute_path: bool        = False,
) -> list[Path]:
    """Return configuration files for project and model.

    List configuration files found under project and model config
    directories and return either filenames or absolute paths.

    Args:
        project_root: Project root path.
        model_root: Optional model-specific root to include. Defaults to None.
        model: Optional model name to filter results. Defaults to None.
        absolute_path: If True, return absolute Paths; otherwise return
            names. Defaults to False.

    Returns:
        Sorted list of configuration file paths or filenames.
    """
    def is_valid(x) -> bool:
        return x is not None and str(x).lower() not in ["", "none"]

    def collect_config_files(root: Path) -> list[Path]:
        config_dir = Path(root) / "config"
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
    
    # Gather all potential files
    config_files = []
    if is_valid(project_root):
        config_files.extend(collect_config_files(project_root))
    if is_valid(model_root):
        config_files.extend(collect_config_files(model_root))
    
    # Filter by file type
    # keeps .yaml/.json (via is_config_file) and non-init .py files
    config_files = [
        cf for cf in config_files
        if cf.is_config_file() or (cf.suffix == ".py" and cf.name != "__init__.py")
    ]
    
    # Optional Model Filtering
    if is_valid(model):
        config_files = [cf for cf in config_files if model in cf.name]
    
    # Format Output
    results = [cf if absolute_path else cf.name for cf in config_files]
    
    return sorted(list(set(results)))


def list_datasets(task: str, mode: str, project_root: Path | None = None) -> list[str]:
    """Return dataset names for a task and mode.

    Determine datasets that support the task and required split, and
    apply project defaults if available.

    Args:
        task: Task name.
        mode: Run mode, e.g., "train" or "predict".
        project_root: Optional project root to apply defaults.
            Defaults to None.

    Returns:
        List of dataset names supporting the task and split.
    """
    # Standardize inputs to Enums
    task_enum = Task(task)
    
    # Map execution mode to data split requirements
    if mode == "train":
        required_split = Split.TRAIN
    elif mode == "val":
        required_split = Split.VAL
    else:
        required_split = Split.TEST
    
    # Global Discovery
    # DATASETS is assumed to be a registry of Dataset metadata objects
    datasets = [
        name for name, meta in DATASETS.items()
        if task_enum in meta.tasks and required_split in meta.splits
    ]
    
    # Project-Level Scoping
    if project_root:
        default_configs  = load_project_defaults(project_root)
        allowed_datasets = default_configs.get("DATASETS")
        
        if allowed_datasets:
            # Set lookup is O(1)
            allowed_set = set(allowed_datasets)
            datasets    = [d for d in datasets if d in allowed_set]
    
    return sorted(datasets)


def list_weights_files(model: str, project_root: Path | None = None) -> list[Path]:
    """Return available weight files for a model.

    Search the project runs and the global zoo directory and return
    matching weight files.

    Args:
        model: Model name to filter weights.
        project_root: Optional project root to include run/train outputs.
            Defaults to None.

    Returns:
        Sorted list of matching weight file paths.
    """
    def is_valid_root(r) -> bool:
        return r is not None and str(r).lower() not in ["", "none"]
    
    def collect_weights(root: Path) -> list[Path]:
        if not root.exists():
            return []
        # Optimization: rglob with specific extensions if is_weights_file permits
        # Otherwise, stick to * but ensure it's a file
        return [f for f in root.rglob("*") if f.is_file() and f.is_weights_file()]
    
    all_files = []
    
    # Collect from local training runs
    if is_valid_root(project_root):
        train_dir = Path(project_root) / "run" / "train"
        all_files.extend(collect_weights(train_dir))
        
    # Collect from global Model Zoo
    if is_valid_root(ZOO_DIR):
        all_files.extend(collect_weights(ZOO_DIR))
    
    # Filter by Model Name
    # We check parts to ensure the weight belongs to a folder/file named after the model
    filtered = [
        f.absolute() for f in all_files
        if model.lower() in [p.lower() for p in f.parts]
    ]
    
    # Return sorted unique paths
    return sorted(list(set(filtered)))

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
        verbose: If True, log load success or failure. Defaults to True.

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


def load_project_defaults(project_root: Path | None) -> dict:
    """Load a project's default configuration.

    Read the project's config/default.py and return the defined defaults.

    Args:
        project_root: Project root path.

    Returns:
        Defaults mapping loaded from the project's default.py, or an empty
        mapping if none is present.
    """
    # Validate Input
    if not project_root or str(project_root).lower() == "none":
        return {}
    
    project_root = Path(project_root)
    config_file  = project_root / "config" / "default.py"
    
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

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def parse_data_dir(root: Path | None, data_dir: Path | str = "") -> Path:
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
            target,                       # Direct path
            root_path / target,           # Relative to project root
            root_path / "data" / target,  # Inside project data folder
            ROOT_DIR  / "data" / target   # Inside global data folder
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

    raise FileNotFoundError(f"Could not resolve data directory. "
                            f"Looked in: {[str(c) for c in candidates]}")


def parse_model_dir(arch: str, model: str) -> Optional[Path]:
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


def parse_save_dir(
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


def parse_weights_dir(root: Path, weights: Path | Sequence[Path]) -> Path | Sequence[Path]:
    """Resolve weight directories from root and weight names.

    Convert relative weight names to absolute directories under the
    project root or the global ROOT_DIR.

    Args:
        root: Project root path.
        weights: Weight name or iterable of weight names.

    Returns:
        Resolved path, a list of resolved paths, or None if nothing was found.
    """
    root = Path(root).normalize(exist=True)
    # Ensure weights is always a list of Path objects
    weight_items = [Path(w) for w in to_list(weights) if w not in [None, "None", ""]]
    
    resolved = []
    for w in weight_items:
        # Check if the weight provided is already an absolute path
        if w.is_absolute() and w.is_dir():
            resolved.append(w)
            continue
            
        # Check Local Project Root (Highest Priority)
        local_dir = root / w
        if local_dir.is_dir():
            resolved.append(local_dir)
            continue
            
        # Check Global Zoo Directory
        global_dir = ROOT_DIR / w
        if global_dir.is_dir():
            resolved.append(global_dir)
            continue
            
    # Return formatted output based on quantity found
    if not resolved:
        return None
    return resolved[0] if len(resolved) == 1 else resolved


def parse_config_file(config: Path, project_root: Path, model_root: Path | None = None) -> Path | None:
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


def parse_weights_file(root: Path, weights: Path | Sequence[Path]) -> Path | Sequence[Path]:
    """Resolve weight file paths given root and weight names.

    Convert weight names to existing weight files under the project root
    or the global ROOT_DIR.

    Args:
        root: Project root path.
        weights: Weight file name or iterable of names.

    Returns:
        Resolved path, a list of resolved paths, or None if nothing was found.
    """
    root = Path(root).normalize(exist=True)
    # Convert input to a standardized list of Path objects
    weight_items = [Path(w) for w in to_list(weights) if w not in [None, "None", ""]]
    
    resolved = []
    for w in weight_items:
        # Handle Absolute Paths
        if w.is_absolute() and w.is_file():
            resolved.append(w)
            continue
            
        # Search Local Project Root (Priority)
        # Search specifically for the file in the project's training runs
        local_file = root / w
        if local_file.is_file() and local_file.is_weights_file():
            resolved.append(local_file)
            continue
            
        # Search Global Model Zoo
        global_file = ROOT_DIR / w
        if global_file.is_file() and global_file.is_weights_file():
            resolved.append(global_file)
            continue
            
    # Return formatted output
    if not resolved:
        return None
    return resolved[0] if len(resolved) == 1 else resolved


def parse_weights_from_config(config: Path | dict) -> Path | None:
    """Extract a weights path from a config file or mapping.

    Inspect the provided config and return the configured weights path
    if present.

    Args:
        config: Path to config or dict-like config.

    Returns:
        Weights path if present, otherwise None.
    """
    if config is None:
        return None
    
    # Handle if config is already a dictionary/Box
    if isinstance(config, (dict, box.Box)):
        weights = config.get("weights")
        return Path(weights) if weights else None

    # Handle if config is a path to a file
    config_path = Path(config)
    if not config_path.is_file():
        return None

    # Load the config (suppressing logs for this utility check)
    args        = load_config(config_path, verbose=False)
    weights_val = args.get("weights")
    
    if not weights_val:
        return None
        
    weights_path = Path(weights_val)
    
    # Smart Resolution: If the weights path is relative,
    # check if it's relative to the config file's directory.
    if not weights_path.is_absolute():
        nearby_weights = config_path.parent / weights_path
        if nearby_weights.exists():
            return nearby_weights
            
    return weights_path

# endregion


# ==============================================================================
# region DEBUGGING
# ==============================================================================

# --- Basic Logging ---

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
