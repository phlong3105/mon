#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Filesystem.

This module provides filesystem utilities.
"""

from __future__ import annotations

__all__ = [
    "delete_files",
    "download_url_to_file",
    "parse_model_fullname",
    "resolve_config_file",
    "resolve_data_dir",
    "resolve_dataset_dir",
    "resolve_output_dir",
    "resolve_project_root",
    "resolve_save_dir",
    "resolve_weights_dir",
    "resolve_weights_file",
]

import requests

from .console import log, log_error
from .constants import K
from .path import Path
from .ui import create_download_bar
from .utils import depascalize, is_valid_str


# ==============================================================================
# region FILESYSTEM
# ==============================================================================

def delete_files(path: Path, regex: str = "", recursive: bool = False):
    """Delete files matching a pattern under a given path.

    Args:
        path (Path): File or directory path to search under.
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
                log(f"path is a directory, to delete, use `path.rmdir()`.")
        except Exception as err:
            log_error(f"could not delete {path.as_posix()}: {err}")
        return

    # If a pattern is given, search for matching files and delete them.
    root = path if path.is_dir() else path.parent
    files = root.rglob(regex) if recursive else root.glob(regex)
    for f in files:
        try:
            if f.is_file():
                f.unlink()
        except Exception as err:
            log_error(f"failed to delete {f}: {err}")


def download_url_to_file(url: Path, path: Path, overwrite: bool = False) -> Path:
    """Download a file from a URL to the local filesystem.

    Args:
        url (Path): URL to download the file from.
        path (Path): Destination path to save the file.
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
        raise ValueError(f"expected a valid URL, got {url.as_posix()}")

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
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

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


def resolve_project_root(cwd: Path) -> Path | None:
    """Resolve the absolute path to the project root directory.

    Args:
        cwd (Path): Current working directory to resolve from. This can be any
            location within the project.

    Returns:
        Path | None: Path to the project root directory if found, otherwise None.
    """
    # Normalize inputs
    cwd = Path(cwd).normalize()

    # Look for pyproject.toml in parent directories
    for parent in cwd.parents:
        if (parent / "pyproject.toml").exists():
            return parent.normalize()

    return None


def resolve_data_dir(cwd: Path) -> Path | None:
    """Resolve the absolute path to the directory containing all datasets in
    the current project.

    Args:
        cwd (Path): Current working directory to resolve from. This can be any
            location within the project.

    Returns:
        Path | None: Path to the dataset directory if found, otherwise None.

    Raises:
        ValueError: If neither ``data_root`` nor ``cwd`` are provided.
    """
    # Normalize inputs
    cwd: Path = Path(cwd).normalize()
    if not cwd.has_name("data", exists=True):
        cwd = resolve_project_root(cwd)
        cwd = cwd / "data"

    return cwd


def resolve_dataset_dir(dataset_name: str, data_root: Path) -> Path | None:
    """Resolve the absolute path to one specific dataset.

    Args:
        dataset_name (str): Specific dataset name.
        data_root (Path): The absolute path to all datasets in the current
            project. If the given path is invalid, it will trigger resolution
            to the project root directory and then resolve to the 'data' subdirectory.

    Returns:
        Path | None: Path to the dataset directory if found, otherwise the
            "data" dir in the project.

    Raises:
        ValueError: If neither ``data_root`` nor ``cwd`` are provided.
    """
    # Normalize inputs
    data_root: Path = resolve_data_dir(data_root)

    # Resolve the dataset root
    dataset_dir = data_root / dataset_name
    if dataset_dir.is_dir():
        return dataset_dir

    if data_root.is_dir():
        return data_root

    return None


def resolve_config_file(
    config: Path,
    root: Path,
    model_dir: Path | None = None,
) -> Path | None:
    """Resolve the absolute path to a config file.

    Search project and model config directories and return the first matching
    config file if found.

    Args:
        config (Path): Config's filename or path.
        root (Path): Project root directory.
        model_dir (Path | None, optional): Model root directory. Defaults to None.

    Returns:
        Path | None: Path to the config file if found, otherwise None.
    """
    # Validate inputs
    if not is_valid_str(config):
        return None

    # Normalize inputs
    config_path = Path(config).normalize()

    # Direct path check (if the user provided a valid absolute/relative path)
    if config_path.is_config_file(exists=True):
        return config_path

    # Define search hierarchy (model-specific first, then project-wide)
    search_dirs = []
    if model_dir:
        model_dir = Path(model_dir).normalize()
        search_dirs.append(Path(model_dir) / "configs")
    if root:
        root = Path(root).normalize()
        search_dirs.append(Path(root) / "configs")

    # Search loop
    for d in search_dirs:
        if not d.is_dir():
            continue
        # Check the root of the config dir, then all subdirectories
        # We search for the exact name or the name with common config suffixes
        for candidate in d.rglob("*"):
            if candidate.is_config_file(exists=True):
                # Check if it matches the name or the stem (if no suffix was provided)
                if config_path.stem == candidate.stem:
                    return candidate

    # Failure State
    return None


def resolve_weights_dir(root: Path, weights_path: Path | None) -> Path | None:
    """Resolve the weight directory from the project root and weights' name or
    relative path.

    Args:
        root (Path): Project root directory.
        weights_path (Path | None): Weights' file or directory.

    Returns:
        Path | None: Path to the weight directory if found, otherwise None.
    """
    # Ensure weights is always a Path object
    if is_valid_str(weights_path):
        weights_path = Path(weights_path).normalize()
    else:
        return None

    # Check if the weight provided is already a directory
    if weights_path.is_dir():
        return weights_path
    elif weights_path.is_weights_file(exists=True):
        return weights_path.parent

    # Check local project root (Highest priority)
    local_dir = Path(root) / weights_path
    if local_dir.is_dir():
        return local_dir

    # Check global zoo directory
    global_dir = K.ZOO_ROOT / weights_path
    if global_dir.is_dir():
        return global_dir

    return None


def resolve_weights_file(root: Path, weights_file: Path | None) -> Path | None:
    """Resolve the weight file from the project root and weights' name or
    relative path.

    Args:
        root (Path): Project root directory.
        weights_file (Path | None): Weights' filename or path.

    Returns:
        Path | None: Path to the weight file if found, otherwise None.
    """
    # Ensure weights is always a Path object
    if is_valid_str(weights_file):
        weights_file = Path(weights_file).normalize()
    else:
        return None

    # Check if the weight provided is already a file
    if weights_file.is_weights_file(exists=True):
        return weights_file

    # Check local project root (Highest priority)
    local_file = Path(root) / weights_file
    if local_file.is_weights_file(exists=True):
        return local_file

    # Check global zoo directory
    global_file = K.ZOO_ROOT / weights_file
    if global_file.is_weights_file(exists=True):
        return global_file

    return None


def resolve_output_dir(
    root: Path,
    dirname: str = "",
    arch: str = "",
    model: str = "",
    data: Path | None = None,
) -> Path:
    """Construct an output directory path based on the project root and
    optional components.

    The path is constructed in the following order:
        ``<root>``/``<dirname>``/``<arch>``/``<model>``/``<data>/``

    Each component is only appended if it is a valid string.

    Args:
        root (Path): Project root directory.
        dirname (str, optional): Directory name to append to the output path.
            Defaults to "".
        arch (str, optional): Architecture name to append to the output path.
            Defaults to "".
        model (str, optional): Model name to append to the output path.
            Defaults to "".
        data (Path | None, optional): Dataset name to append to the output path.
            Defaults to None.

    Returns:
        Path: Constructed save directory path.
    """
    # Start with the base root (e.g., 'project/')
    output_dir = Path(root).normalize()
    # Append dirname (e.g., 'runs/train', 'runs/predict')
    output_dir = output_dir.append(dirname)
    # Append architecture and model (e.g., 'yolov8/yolov8n/')
    # output_dir = output_dir.append(f"{arch}{os.sep}{model}")
    output_dir = output_dir.append(arch)
    output_dir = output_dir / model
    # Append dataset (e.g., 'coco128/')
    output_dir = output_dir.append(data)
    # Return the final path
    return output_dir


def resolve_save_dir(
    output_dir: Path,
    dirname: str,
    subdirname: str = "",
    src_path: Path | None = None,
    keep_subdirs: bool = False,
    near_src: bool = False,
) -> Path:
    """Compute the saving directory for an output type based on the output
    directory and optional components.

    The path is computed in the following order:
        ``<output_dir>``/``<dirname>``/``<sub_dirname>``/

    Examples:
        >>> output_dir = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm"
        >>> dirname = "pred"
        >>> subdirname = ""
        >>> src_path = "/Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/image/01.jpg"
        >>> print(resolve_save_dir(output_dir, dirname, subdirname, src_path, False, False).as_posix())
        /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm/pred
        >>> print(resolve_save_dir(output_dir, dirname, subdirname, src_path, True, False).as_posix())
        /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/run/predict/zerodce/zerodce/dicm/test/image
        >>> print(resolve_save_dir(output_dir, dirname, subdirname, src_path, False, True).as_posix())
       /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/pred
        >>> print(resolve_save_dir(output_dir, dirname, subdirname, src_path, True, True).as_posix())
        /Volumes/ssd_01/10_workspace/11_code/mon/projects/enhance/data/dicm/test/image_zerodce

    Args:
        output_dir (Path): Path to the output directory.
        dirname (str): Directory name to append to the output path
            (e.g., 'pred', 'debug').
        subdirname (str, optional): Subdirectory name to append to the output
            path (e.g., 'debug'/'mask'). Defaults to "".
        src_path (Path | None, optional): Source path to determine the
            subdirectory hierarchy. Defaults to None.
        keep_subdirs (bool, optional): If True, preserve the subdirectory
            structure of ``src_path`` relative to ``dirname``. Defaults to False.
        near_src (bool, optional): If True, change the ``output_dir`` to the
            same level as ``src_path``. Defaults to False.

    Returns:
        Path: Computed output directory path.
    """
    # 1. Normalize inputs
    output_dir = Path(output_dir).normalize()
    dirname = Path(dirname)
    subdirname = Path(subdirname) if is_valid_str(subdirname) else ""
    src_path = Path(src_path).normalize() if is_valid_str(src_path) else None

    # 2. Save near source location if requested
    if near_src and src_path:
        if keep_subdirs:
            if output_dir.name != dirname.name:
                root_suffix = output_dir.parent.name
            else:
                root_suffix = output_dir.name
            output_dir = src_path.parent.parent
            return output_dir.append(f"{src_path.parent.name}_{root_suffix}")
        else:
            output_dir = src_path.parent.parent
            return output_dir.append(dirname)

    # 3. Preserve subdirectory structure if requested
    if keep_subdirs and src_path:
        data_name = output_dir.name
        rel_path = src_path.relative_path_to(data_name)
        return output_dir.append(rel_path.parent)

    # 4. Otherwise, return the default save directory
    output_dir = output_dir.append(dirname)
    if subdirname:
        return output_dir.append(subdirname)
    else:
        return output_dir


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
