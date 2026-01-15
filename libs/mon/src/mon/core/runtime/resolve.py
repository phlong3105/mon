#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runtime project resolution utilities.

This module provides functionalities for resolving the project's configuration,
including discovery, validation, and retrieval.
"""

from __future__ import annotations

__all__ = [
    "ProjectResolver",
    "load_config",
    "parse_config_file",
    "parse_data_dir",
    "parse_model_dir",
    "parse_model_fullname",
    "parse_output_dir",
    "parse_save_dir",
    "parse_weights",
    "parse_weights_dir",
    "parse_weights_file",
]

import importlib.util
from typing import Any, Optional

import box
import yaml

from mon.core.console import log, log_error
from mon.core.constants import MONO_ROOT_DIR, ROOT_DIR, ZOO_DIR
from mon.core.dtypes import Weights
from mon.core.enum import Task
from mon.core.factory import DATASETS, MODELS, WEIGHTS
from mon.core.pathlib import Path
from mon.core.utils import depascalize


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
# region VALIDATION
# ==============================================================================


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


def parse_weights_dir(root: Path, weights: Path | str) -> Path | None:
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


def parse_config_file(
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


def parse_weights_file(root: Path, weights: Path) -> Path | None:
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


def parse_weights(
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
    weights = parse_weights_file(root=root, weights=weights)

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
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
