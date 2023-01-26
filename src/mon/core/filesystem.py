#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides facilities for performing operations on file systems
and their components.
"""

from __future__ import annotations

__all__ = [
    "clear_cache", "copy_file_to", "create_dirs", "delete_files", "get_hash",
    "get_latest_file", "get_next_file_version", "has_subdir", "list_files",
    "list_subdirs",
]

import glob
import os
import shutil

from mon.core import builtins, pathlib, rich
from mon.core.typing import PathsType, PathType, Strs


# region Filesystem

def clear_cache(path: PathType, recursive: bool = True):
    r"""Clears cache files in a directory and subdirectories.
    
    Args:
        path: The directory path.
        recursive: If True, recursively look for cache files in subdirectories.
            Defaults to True.
    """
    delete_files(files=path, extension=".cache", recursive=recursive)


def copy_file_to(file: PathType, dst: PathType):
    """Copy a file to a new directory.
    
    Args:
        file: The path to the original file.
        dst: The destination directory.
    """
    create_dirs(paths=[dst])
    file = pathlib.Path(file)
    shutil.copyfile(file, dst / file.name)


def create_dirs(paths: PathsType, recreate: bool = False):
    """Create a list of directories.
    
    Args:
        paths: A list of directories' absolute paths.
        recreate: If True, delete and create existing directories. Defaults to
            False.
    """
    paths = builtins.to_list(paths)
    paths = [pathlib.Path(p) for p in paths if p is not None]
    paths = builtins.unique(paths)
    try:
        for path in paths:
            is_dir = path.is_dir()
            if is_dir and recreate:
                shutil.rmtree(path)
            if not is_dir:
                path.mkdir(parents=True, exist_ok=recreate)
        return 0
    except Exception as err:
        rich.console.log(f"Cannot create directory: {err}.")


def delete_files(
    files    : PathsType = "",
    dirs     : PathsType = "",
    extension: str       = "",
    recursive: bool      = True
):
    """Delete files in directories recursively.
    
    Args:
        files: A list of deleting files.
        dirs: A list of directories to search for the :param:`files`.
        extension: A specific file extension. Defaults to ““.
        recursive: If True, recursively look for the :param:`files` in
            subdirectories. Defaults to True.
    """
    files     = builtins.to_list(files)
    files     = [pathlib.Path(f) for f in files if f is not None]
    files     = [f for f in files if f.is_file()]
    files     = builtins.unique(files)
    dirs      = builtins.to_list(dirs)
    dirs      = [pathlib.Path(d) for d in dirs if d is not None]
    dirs      = builtins.unique(dirs)
    extension = f".{extension}" if "." not in extension else extension
    for d in dirs:
        if recursive:
            files += list(d.rglob(*{extension}))
        else:
            files += list(d.glob(extension))
    for f in files:
        rich.console.log(f"Deleting {f}.")
        f.unlink()


def get_hash(files: PathsType) -> int:
    """Get the total size (in bytes) of all files."""
    files = builtins.to_list(files)
    files = [pathlib.Path(f) for f in files if f is not None]
    return sum(f.stat().st_size for f in files if f.is_file())


def get_latest_file(path: PathType) -> pathlib.Path | None:
    """Get the latest file, in other words, the file with the latest created
    time, in a directory.
    
    Args:
        path: A directory path.
    """
    if path is not None:
        file_list = list(pathlib.Path(path).rglob("*"))
        if len(file_list) > 0:
            return max(file_list, key=os.path.getctime)
    return None


def get_next_file_version(root_dir: PathType, prefix: str | None = None) -> int:
    """Get the next file version number in a directory.
    
    Args:
        root_dir: The directory path containing files with :param:`prefix`.
        prefix: The file prefix.
    """
    try:
        listdir_info = os.listdir(str(root_dir))
    except OSError:
        return 0
    
    existing_versions = []
    for listing in listdir_info:
        if isinstance(listing, str):
            d = listing
        elif isinstance(listing, dict):
            d = listing["name"]
        else:
            d = ""
        bn = os.path.basename(d)
        if bn.startswith("version-") \
            or bn.startswith("exp-") \
            or (isinstance(prefix, str) and bn.startswith(prefix + "-")):
            dir_ver = bn.split("-")[-1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    
    return max(existing_versions) + 1


def has_subdir(path: PathType, name: str) -> bool:
    """Return True if a directory has a subdirectory with a given name.
    
    Args:
        path: A path to the directory.
        name: The name of the subdirectory to search for.
    """
    return name in list_subdirs(path)


def list_files(patterns: Strs) -> list[PathType]:
    """List all files matching given patterns.
    
    Args:
        patterns: A list of file patterns.
    
    Returns:
        A list of unique file paths.
    """
    patterns    = builtins.to_list(patterns)
    patterns    = [p for p in patterns if p is not None]
    patterns    = builtins.unique(patterns)
    image_paths = []
    for pattern in patterns:
        for abs_path in glob.glob(pattern):
            if os.path.isfile(abs_path):
                image_paths.append(pathlib.Path(abs_path))
    return builtins.unique(image_paths)


def list_subdirs(path: PathType | None) -> list[PathType] | None:
    """List all subdirectories inside a directory.
    
    Args:
        path: A directory path.
    
    Returns:
        A list of subdirectories' paths.
    """
    if path is None:
        return None
    path = str(pathlib.Path)
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# endregion
