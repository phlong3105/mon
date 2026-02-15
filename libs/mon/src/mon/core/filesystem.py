#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Filesystem.

This module provides filesystem utilities.
"""

from __future__ import annotations

__all__ = [
    "delete_files",
    "download_url_to_file",
    "list_weights_files",
    "parse_model_fullname",
    "resolve_config_file",
    "resolve_data_root",
    "resolve_model_dir",
    "resolve_output_dir",
    "resolve_save_dir",
    "resolve_weights",
    "resolve_weights_dir",
    "resolve_weights_file",
]

import requests

from mon.core.constants import MODELS, MONO_ROOT, ROOT, WEIGHTS, ZOO_ROOT
from mon.core.data import Weights
from mon.core.logger import log, log_error
from mon.core.path import Path
from mon.core.typing import PathLike
from mon.core.ui import create_download_bar
from mon.core.utils import depascalize, is_valid_str


# ==============================================================================
# region FILESYSTEM
# ==============================================================================

def delete_files(path: PathLike, regex: str = "", recursive: bool = False):
    """Delete files matching a pattern under a given path.

    Args:
        path (PathLike): File or directory path to search under.
        regex (str, optional): Glob pattern to match files. Defaults to ".
        recursive (bool, optional): If True, search subdirectories recursively.
            Defaults to False.
    """
    path = Path(path).normalize()

    # If no pattern, delete the single path if it's a file.
    if not regex:
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                # Safety check: Do not delete directories without a pattern.
                log(f"Path is a directory. To delete, use `path.rmdir()`.")
        except Exception as err:
            log_error(f"Could not delete {path}: {err}")
        return

    # If a pattern is given, search for matching files and delete them.
    root = path if path.is_dir() else path.parent
    files = root.rglob(regex) if recursive else root.glob(regex)
    for f in files:
        try:
            if f.is_file():
                f.unlink()
        except Exception as err:
            log_error(f"Failed to delete {f}: {err}")


def download_url_to_file(
    url: PathLike,
    path: PathLike,
    overwrite: bool = False
) -> Path:
    """Download a file from a URL to the local filesystem.

    Args:
        url (PathLike): URL to download the file from.
        path (PathLike): Destination path to save the file.
        overwrite (bool, optional): If True, overwrite the destination file if
            it exists. Defaults to False.

    Raises:
        ValueError: If ``url`` is not a valid URL.
        requests.HTTPError: If the download fails.
    """
    path = Path(path).normalize()
    if path.exists() and not overwrite:
        return path

    # Check URL
    if not Path(url).is_url():
        raise ValueError(f"Expected a valid URL, but got '{url}'.")

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Download file in chunks
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()  # Raise an exception for bad status codes
    total_size = int(response.headers.get("content-length", 0))

    with create_download_bar() as pbar:
        task_id = pbar.add_task(f"[cyan]Downloading {path.name}", total=total_size)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(task_id, advance=len(chunk))

    return path

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================

def list_weights_files(dir_path: PathLike) -> list[Path]:
    """List all weights files in the given ``root`` directory.

    Args:
        dir_path (PathLike): Directory path to search for weights files.

    Returns:
        list[Path]: List of paths to weights files.
    """
    dir_path = Path(dir_path).normalize()
    if not dir_path.exists():
        return []

    # Optimization: rglob with specific extensions if is_weights_file permits
    # Otherwise, stick to * but ensure it's a file
    return [f for f in dir_path.rglob("*") if f.is_weights_file(exist=True)]

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def resolve_config_file(
    config: PathLike,
    project_dir: PathLike,
    model_dir: PathLike | None = None,
) -> Path | None:
    """Resolve the absolute path to a config file.

    Search project and model config directories and return the first
    matching config file if found.

    Args:
        config (PathLike): Config name or path.
        project_dir (PathLike): Project root directory.
        model_dir (PathLike, optional): Model root directory. Defaults to None.

    Returns:
        Path: Resolved config file path if found, otherwise None.
    """
    # Validate inputs
    if not is_valid_str(config):
        return None

    config_path = Path(config).normalize()

    # Direct path check (if the user provided a valid absolute/relative path)
    if config_path.exists() and config_path.is_file():
        return config_path

    # Define search hierarchy (model-specific first, then project-wide)
    search_roots = []
    if model_dir:
        search_roots.append(Path(model_dir) / "config")
    if project_dir:
        search_roots.append(Path(project_dir) / "config")

    # Search loop
    for root in search_roots:
        if not root.is_dir():
            continue

        # Check the root of the config dir, then all subdirectories
        # We search for the exact name or the name with common config suffixes
        for candidate in root.rglob("*"):
            if candidate.is_config_file(exist=True):
                # Check if it matches the name or the stem (if no suffix was provided)
                if config_path.stem == candidate.stem:
                    return candidate

    # Failure State
    log_error(f"Config not found: {config}. Searched in {search_roots}")
    return None


def resolve_data_root(root: PathLike | None, data_dir: PathLike = "") -> Path:
    """Resolve an absolute data directory path from candidates.

    Try a series of candidate locations and return the first existing
    directory. Raise an error if no candidate exists.

    Args:
        root (PathLike): Project root directory. Defaults to None.
        data_dir (PathLike, optional): Data directory name or path. Defaults to "".

    Returns:
        Path: Resolved absolute data directory path.

    Raises:
        FileNotFoundError: If no candidate data directory is found.
    """
    # Use global ROOT_DIR if no project root is provided
    root_path = Path(root).normalize() if root else ROOT

    # Identify the target name/path
    target = Path(data_dir).normalize() if data_dir else None

    # Build Ordered Candidates
    candidates = []

    if target:
        # If target is absolute, Path logic will prioritize it during joins
        candidates.extend(
            [
                target,                            # Direct path
                root_path / target,            # Relative to project root
                root_path / "data" / target,   # Inside project data folder
                ROOT / "data" / target,   # Inside "mon" data folder
                MONO_ROOT / "data" / target,   # Inside global data folder (monorepo)
            ]
        )

    # Fallback search locations
    candidates.extend(
        [
            root_path / "data",
            ROOT / "data",
        ],
    )

    # Validation Loop
    # Use unique paths only to avoid multiple disk IO checks on the same location
    seen = set()
    for d in candidates:
        d = d.nomralize()
        if d not in seen and d.is_dir():
            return d
        seen.add(d)

    raise FileNotFoundError(
        f"Could not resolve data directory. "
        f"Looked in: {[str(c) for c in candidates]}."
    )


def parse_model_fullname(name: str, data: str = "", suffix: str = "") -> str:
    """Compose a model fullname from name, data, and optional suffix.

    Args:
        name (str): Base architecture/model name.
        data (str, optional): Dataset identifier. Defaults to "".
        suffix (str, optional): Optional suffix (e.g., 'nano', 'v2').
            Defaults to "".

    Returns:
        str: Full model name.
    """
    if not is_valid_str(name):
        # Using a default or raising is often better than just logging
        return "unnamed_model"

    # Start with the base architecture/model name
    fullname = str(name).strip()

    # Append dataset identifier
    if is_valid_str(data):
        data = depascalize(str(data).strip())
        if data not in fullname:
            fullname = f"{fullname}_{data}"

    # Append optional suffix
    if is_valid_str(suffix):
        suffix = depascalize(str(suffix).strip())
        if suffix not in fullname:
            fullname = f"{fullname}_{suffix}"

    return fullname


def resolve_model_dir(arch: str, model: str) -> Path | None:
    """Return the model directory for the given arch and model.

    Args:
        arch (str): Architecture name.
        model (str): Model name.

    Returns:
        Path: Model directory path if found, otherwise None.
    """
    # Validate inputs
    if not arch or not model:
        return None

    # Look up the model directory in the registry
    try:
        # Access nested registry.
        # Using .get() allows for a more graceful failure than raw brackets.
        arch_entry = MODELS.get(arch)
        if arch_entry is None:
            return None

        model_entry = arch_entry.get(model)
        if model_entry is None:
            return None

        # Path resolution
        model_dir = model_entry.get("model_dir")
        if model_dir:
            return Path(model_dir)

    except Exception as e:
        # If logging is available, log the registry access failure
        return None

    return None


def resolve_save_dir(
    root: PathLike,
    arch: str = "",
    model: str = "",
    data: str = "",
) -> Path:
    """Build a save directory path from components.

    Combine root, architecture, model and optional data to construct a
    save directory path suitable for storing run outputs.

    Args:
        root (PathLike): Project root directory.
        arch (str, optional): Architecture name. Defaults to "".
        model (str, optional): Model name. Defaults to "".
        data (str, optional): Dataset name. Defaults to "".

    Returns:
        Path: Resolved save directory path.
    """
    # Start with the base root (e.g., 'project/runs/train')
    save_dir = Path(root).normalize()

    # Add architecture level (e.g., 'yolov8')
    if is_valid_str(arch):
        save_dir /= depascalize(str(arch).strip())

    # Add model level (e.g., 'yolov8n')
    if is_valid_str(model):
        save_dir /= depascalize(str(model).strip())

    # Add dataset level inside the model folder
    if is_valid_str(data):
        data_path = Path(data)
        # If it's a real path, take the filename (stem); otherwise, use the string directly
        folder_name = data_path.stem if (data_path.suffix or data_path.exists()) else str(data)
        save_dir /= depascalize(str(folder_name).strip())

    return save_dir


def resolve_output_dir(
    root: PathLike,
    dirname: PathLike,
    subdir: PathLike,
    src_path: PathLike | None = None,
    keep_subdirs: bool = False,
) -> Path:
    """Compute the output directory for a source path.

    Determine where to place outputs for a given source path, optionally
    preserving subdirectory structure or saving outputs near the source.

    Args:
        root (PathLike): Project root directory.
        dirname (PathLike): Directory under root to place outputs.
        subdir (PathLike): Subdirectory under dirname to place outputs.
        src_path (PathLike, optional): Source path to determine the
            subdirectory hierarchy. Defaults to None.
        keep_subdirs (bool, optional): If True, preserve the subdirectory
            structure of ``src_path`` relative to ``dirname``. Defaults to False.

    Returns:
        Path: Resolved output directory path.
    """
    root = Path(root).normalize()
    dirname = Path(dirname)
    subdir = Path(subdir) if is_valid_str(subdir) else None
    src_path = Path(src_path).normalize() if is_valid_str(src_path) else None

    # Preserve subdirectory structure if requested
    if keep_subdirs and src_path:
        try:
            # Get path relative to the input root (dirname)
            # e.g., src: 'data/val/class1/img.jpg', dir: 'data' -> 'val/class1'
            rel_path = src_path.parent.relative_to(dirname)
            target_path = root / rel_path
        except ValueError:
            # Fallback if src_path is not under dirname
            target_path = root / src_path.parent.name

        if subdir:
            return target_path / subdir.stem
        return target_path

    # Default behavior: just use root + dirname stem + optional subdir
    final_root = root
    if dirname.stem != root.stem:
        final_root = root / dirname.stem
    if subdir:
        return final_root / subdir.stem
    return final_root


def resolve_weights_dir(root: PathLike, weights: PathLike) -> Path | None:
    """Resolve the weight directory from the given root and weights name or
    relative path.

    Args:
        root (PathLike): Project root directory.
        weights (PathLike): Weights name or relative path.

    Returns:
        Path: Absolute weights directory path or None if nothing was found.
    """
    root = Path(root).normalize()
    # Ensure weights is always a Path object
    weights = Path(weights) if is_valid_str(weights) else None

    # Check if the weight provided is already an absolute path
    if weights.is_absolute() and weights.is_dir():
        return weights

    # Check local project root (Highest priority)
    local_dir = root / weights
    if local_dir.is_dir():
        return local_dir

    # Check global zoo directory
    global_dir = ZOO_ROOT / weights
    if global_dir.is_dir():
        return global_dir

    # Return None if not found
    return None


def resolve_weights_file(root: PathLike, weights: PathLike) -> Path | None:
    """Resolve the weight file from the given root and weights name or
    relative path.

    Args:
        root (PathLike): Project root directory.
        weights (PathLike): Weights name or relative path.

    Returns:
        Path: Absolute weight file path or None if nothing was found.
    """
    root = Path(root).normalize()
    # Ensure weights is always a Path object
    weights = Path(weights) if is_valid_str(weights) else None

    # Check if the weight provided is already an absolute path
    if weights.is_absolute() and weights.is_weights_file():
        return weights

    # Check local project root (Highest priority)
    # Search specifically for the file in the project's training runs
    local_file = root / weights
    if local_file.is_weights_file(exist=True):
        return local_file

    # Check global zoo directory
    from mon.core.constants import ZOO_ROOT
    global_file = ZOO_ROOT / weights
    if global_file.is_weights_file(exist=True):
        return weights

    # Return None if not found
    return None


def resolve_weights(
    root: PathLike,
    weights: PathLike,
    num_classes: int | None = None,
) -> Weights | None:
    """Resolve a ``Weights`` object from the given root and weights name or
    relative path.

    Args:
        root (PathLike): Project root directory.
        weights (PathLike): Weights name or relative path.
        num_classes (int, optional): Number of classes to set in the ``Weights``
            object if found. Defaults to None.

    Returns:
        Weights: ``Weights`` object if found, otherwise None.
    """
    weights = resolve_weights_file(root=root, weights=weights)

    # If a valid weights file was found, wrap it in a Weights object
    if weights:
        # Check if the weights object is already registered in WEIGHTS
        if WEIGHTS.has(weights_path=weights):
            return WEIGHTS.find_weights_objs(weights_path=weights)
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
