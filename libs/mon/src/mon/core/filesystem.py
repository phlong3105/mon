#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Filesystem utilities.

This module provides filesystem-related utilities.
"""

from __future__ import annotations

__all__ = [
    "delete_files",
    "download_url_to_file",
    "parse_model_fullname",
    "resolve_config_file",
    "resolve_data_dir",
    "resolve_model_dir",
    "resolve_output_dir",
    "resolve_save_dir",
]

from typing import Optional

import requests

from mon.core.console import log, log_error
from mon.core.constants import MONO_ROOT_DIR, ROOT_DIR
from mon.core.pathlib import Path
from mon.core.utils import depascalize, is_valid_str


# ==============================================================================
# region FILESYSTEM
# ==============================================================================

def delete_files(path: str | Path, regex: str = None, recursive: bool = False):
    """Delete files matching a pattern under a given path.

    Args:
        path: Path or directory to delete from.
        regex: Glob pattern to match files. Defaults to None.
        recursive: If True, search recursively for matches. Defaults to False.
    """
    path = Path(path)

    if not regex:
        # If no pattern, delete the single path if it's a file.
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
    search_root     = path if path.is_dir() else path.parent
    files_to_delete = search_root.rglob(regex) if recursive else search_root.glob(regex)

    for f in files_to_delete:
        try:
            if f.is_file():
                f.unlink()
        except Exception as err:
            log_error(f"Failed to delete {f}: {err}")


def download_url_to_file(url: str, path: str | Path, overwrite: bool = False) -> Path:
    """Download a file from a URL to the local filesystem.

    Args:
        url: Source URL to download from.
        path: Destination path to save the file.
        overwrite: If True, overwrite the destination file if it exists.
            Defaults to False.

    Raises:
        ValueError: If ``url`` is not a valid URL.
        requests.HTTPError: If the download fails.
    """
    dest_path = Path(path)
    if dest_path.exists() and not overwrite:
        return dest_path

    if not Path(url).is_url():
        raise ValueError(f"Expected a valid URL, but got '{url}'.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Import rich locally to avoid circular dependencies and keep it optional.
    from mon.core.rich import create_download_bar

    response   = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()  # Raise an exception for bad status codes
    total_size = int(response.headers.get("content-length", 0))

    with create_download_bar() as pbar:
        task_id = pbar.add_task(f"[cyan]Downloading {dest_path.name}", total=total_size)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(task_id, advance=len(chunk))
    return dest_path

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


def resolve_data_dir(root: Path | None, data_dir: Path | str = "") -> Path:
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


def resolve_model_dir(arch: str, model: str) -> Optional[Path]:
    """Return the model directory for the given arch and model.

    Args:
        arch: Architecture name.
        model: Model name.

    Returns:
        Path to the model directory, or None if unspecified.
    """
    from mon.core.factory import MODELS

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
        model_dir = getattr(model_entry, "_model_dir") or getattr(model_entry, "model_dir")

        if model_dir:
            return Path(model_dir)
    except Exception as e:
        # If logging is available, log the registry access failure
        return None

    return None


def resolve_save_dir(
    root : Path | str,
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


def resolve_output_dir(
    root        : Path | str,
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
    subdir_name = str(subdir_name) if is_valid_str(subdir_name) else None
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


def resolve_config_file(
    config      : Path | str,
    project_root: Path | str,
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


# --- Selection ---


# --- Aggregation ---


# endregion
