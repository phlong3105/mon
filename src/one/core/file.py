#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for managing files and directories.

File, dir, Path definitions:
- drive         : A string that represents the drive name. For example,
                  PureWindowsPath("c:/Program Files/CSV").drive returns "C:"
- parts         : Return a tuple that provides access to the path"s components
- name          : Path component without any dir.
- parent        : sequence providing access to the logical ancestors of the
                  path.
- stem          : Final path component without its suffix.
- suffix        : the file extension of the final component.
- anchor        : Part of a path before the dir. / is used to create child
                  paths and mimics the behavior of os.path.join.
- joinpath      : Combine the path with the arguments provided.
- match(pattern): Return True/False, based on matching the path with the
                  glob-style pattern provided

Working with Path:
path = Path("/home/mains/stackabuse/python/sample.md")
- path              : Return PosixPath("/home/mains/stackabuse/python/sample.md")
- path.parts        : Return ("/", "home", "mains", "stackabuse", "python")
- path.name         : Return "sample.md"
- path.stem         : Return "sample"
- path.suffix       : Return ".md"
- path.parent       : Return PosixPath("/home/mains/stackabuse/python")
- path.parent.parent: Return PosixPath("/home/mains/stackabuse")
- path.match("*.md"): Return True
- PurePosixPath("/python").joinpath("edited_version"): returns ("home/mains/stackabuse/python/edited_version)
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Union

import validators

from one.core.collection import unique
from one.core.rich import console
from one.core.types import ImageFormat
from one.core.types import ScalarListOrTupleAnyT
from one.core.types import VideoFormat


# MARK: - Functional

def create_dirs(paths: ScalarListOrTupleAnyT[str], recreate: bool = False):
    """Check and create directories.

    Args:
        paths (ScalarListOrTupleAnyT[str]):
            List of directories' paths to create.
        recreate (bool):
            If `True`, delete and recreate existing directories.
    """
    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, tuple):
        paths = list(paths)
    paths       = [p for p in paths if p is not None]
    unique_dirs = unique(paths)
    try:
        for d in unique_dirs:
            if os.path.exists(d) and recreate:
                shutil.rmtree(d)
            if not os.path.exists(d):
                os.makedirs(d)
        return 0
    except Exception as err:
        console.log(f"Cannot create directory: {err}.")


def delete_files(
    files    : ScalarListOrTupleAnyT[str] = "",
    dirs     : ScalarListOrTupleAnyT[str] = "",
    extension: str  = "",
    recursive: bool = True
):
    """Delete all files in directories that match the desired extension.

    Args:
        files (ScalarListOrTupleAnyT[str]):
            List of files that contains the files to be deleted. Default: "".
        dirs (ScalarListOrTupleAnyT[str]):
            List of directories that contains the files to be deleted.
            Default: "".
        extension (str):
            File extension. Default: "".
        recursive (bool):
            Search subdirectories if any. Default: `True`.
    """
    if isinstance(files, str):
        files = [files]
    elif isinstance(files, tuple):
        files = list(files)
    if isinstance(dirs, str):
        dirs = [dirs]
    elif isinstance(dirs, tuple):
        dirs = list(dirs)
    files     = [f for f in files if os.path.isfile(f)]
    files     = unique(files)
    dirs      = [d for d in dirs if d is not None]
    dirs      = unique(dirs)
    extension = f".{extension}" if "." not in extension else extension
    for d in dirs:
        pattern  = os.path.join(d, f"*{extension}")
        files   += glob(pattern, recursive=recursive)
    for f in files:
        console.log(f"Deleting {f}.")
        os.remove(f)
        

def get_hash(files: ScalarListOrTupleAnyT[str]) -> int:
    """Get a single hash value of a list of files.
    
    Args:
        files (ScalarListOrTupleAnyT[str]):
            List of files.
    
    Returns:
        hash (int):
            Sum of all file size.
    """
    if isinstance(files, str):
        files = [files]
    elif isinstance(files, tuple):
        files = list(files)
    files = [f for f in files if f is not None]
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def get_latest_file(path: str, recursive: bool = True) -> Union[str, None]:
    """Get the latest file from a folder or pattern according to modified time.

    Args:
        path (str):
            Directory path or path pattern.
        recursive (str):
            Should look for sub-directories also?.

    Returns:
        latest_path (str, None):
            Latest file path. Return `None` if not found (no file, wrong path
            format, wrong file extension).
    """
    if path:
        file_list = glob(path, recursive=recursive)
        if len(file_list) > 0:
            return max(file_list, key=os.path.getctime)
    return None


def has_subdir(path: str, name: str) -> bool:
    """Return `True` if the subdirectory with `name` is found inside `path`."""
    return name in list_subdirs(path=path)


def is_basename(path: Union[str, None]) -> bool:
    """Check if the given path is a basename, i.e, a file path without extension.
    """
    if path is None:
        return False
    if str(Path(path).parent) == ".":
        root, ext = os.path.splitext(path)
        if ext != "":
            return True
    return False


def is_bmp_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.bmp` image file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() == ImageFormat.BMP.value:
        return True
    return False


def is_ckpt_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.ckpt` file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["ckpt"]:
        return True
    return False


def is_image_file(path: Union[str, None]) -> bool:
    """Check if the given path is an image file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ImageFormat.values():
        return True
    return False


def is_json_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.json` file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and path.split(".")[1].lower() in ["json"]:
        return True
    return False


def is_name(path: Union[str, None]) -> bool:
    """Check if the given path is a name with extension."""
    if path is None:
        return False
    if path == str(Path(path).stem):
        return True
    return False


def is_stem(path: Union[str, None]) -> bool:
    """Check if the given path is a stem, i.e., a name without extension."""
    if path is None:
        return False
    if str(Path(path).parent) == ".":
        root, ext = path.split(".")
        if ext == "":
            return True
    return False


def is_torch_saved_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.pt`, `.pth`, `.weights`, or `.ckpt` file.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["pt", "pth", "weights", "ckpt"]:
        return True
    return False


def is_txt_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.txt` file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["txt"]:
        return True
    return False


def is_url(path: Union[str, None]) -> bool:
    """Check if the given path is a valid url."""
    if path is None:
        return False
    if isinstance(validators.url(path), validators.ValidationFailure):
        return False
    return True


def is_url_or_file(path: Union[str, None]) -> bool:
    """Check if the given path is a valid url or a local file."""
    if path is None:
        return False
    if isinstance(validators.url(path), validators.ValidationFailure) or \
       os.path.isfile(path=path):
        return False
    return True


def is_video_file(path: Union[str, None]) -> bool:
    """Check if the given path is a video file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in VideoFormat.values():
        return True
    return False


def is_video_stream(path: Union[str, None]) -> bool:
    """Check if the given path is a video stream."""
    if path is None:
        return False
    return "rtsp" in path.lower()


def is_weights_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.pt` or `.pth` file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["pt", "pth"]:
        return True
    return False


def is_xml_file(path: Union[str, None]) -> bool:
    """Check if the given path is a .xml file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["xml"]:
        return True
    return False


def is_yaml_file(path: Union[str, None]) -> bool:
    """Check if the given path is a `.yaml` file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["yaml", "yml"]:
        return True
    return False


def list_files(patterns: ScalarListOrTupleAnyT[str]) -> list[str]:
    """List all files that match the desired patterns.
    
    Args:
        patterns (ScalarListOrTupleAnyT[str]):
            File patterns.
            
    Returns:
        image_paths (list[str]):
            List of images paths.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    elif isinstance(patterns, tuple):
        patterns = list(patterns)
    patterns    = [p for p in patterns if p is not None]
    patterns    = unique(patterns)
    image_paths = []
    for pattern in patterns:
        for abs_path in glob(pattern):
            if os.path.isfile(abs_path):
                image_paths.append(abs_path)
    return unique(image_paths)


def list_subdirs(path: Union[str, None]) -> Union[list[str], None]:
    """List all subdirectories inside the given `path`."""
    if path is None:
        return None
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
