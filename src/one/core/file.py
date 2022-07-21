#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for managing files and directories.

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

import validators

from one.core.rich import console
from one.core.types import ImageFormat
from one.core.types import Strs
from one.core.types import to_list
from one.core.types import unique
from one.core.types import VideoFormat


# MARK: - Functional

def create_dirs(paths: Strs, recreate: bool = False):
    """
    Create a list of directories, if they don't exist.
    
    Args:
        paths (Strs): A list of paths to create.
        recreate (bool): If True, the directory will be deleted and recreated.
            Defaults to False.
    """
    paths       = to_list(paths)
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
    files    : Strs = "",
    dirs     : Strs = "",
    extension: str  = "",
    recursive: bool = True
):
    """
    Deletes files.
    
    Args:
        files (Strs): A list of files to delete.
        dirs (Strs): A list of directories to search for files to delete.
        extension (str): File extension. Defaults to "".
        recursive (bool): If True, then the function will search for files
            recursively. Defaults to True.
    """
    files     = to_list(files)
    files     = [f for f in files if os.path.isfile(f)]
    files     = unique(files)
    dirs      = to_list(dirs)
    dirs      = [d for d in dirs if d is not None]
    dirs      = unique(dirs)
    extension = f".{extension}" if "." not in extension else extension
    for d in dirs:
        pattern  = os.path.join(d, f"*{extension}")
        files   += glob(pattern, recursive=recursive)
    for f in files:
        console.log(f"Deleting {f}.")
        os.remove(f)
        

def get_hash(files: Strs) -> int:
    """
    It returns the sum of the sizes of the files in the list.
    
    Args:
        files (Strs): File paths.
    
    Returns:
        The sum of the sizes of the files in the list.
    """
    files = to_list(files)
    files = [f for f in files if f is not None]
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def get_latest_file(path: str, recursive: bool = True) -> str | None:
    """
    It returns the latest file in a given directory.
    
    Args:
        path (str): The path to the directory you want to search.
        recursive (bool): If True, the pattern “**” will match any files and
            zero or more directories and subdirectories. Defaults to True
    
    Returns:
        The latest file in the path.
    """
    if path:
        file_list = glob(path, recursive=recursive)
        if len(file_list) > 0:
            return max(file_list, key=os.path.getctime)
    return None


def has_subdir(path: Path, name: str) -> bool:
    """
    Return True if the directory at the given path has a subdirectory with the
    given name.
    
    Args:
        path (Path): The path to the directory you want to check.
        name (str): The name of the subdirectory to check for.
    
    Returns:
        A boolean value.
    """
    return name in list_subdirs(path=path)


def is_basename(path: Path | None) -> bool:
    """
    If the path is not None, and the parent of the path is the current
    directory, then the path is a basename.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if str(Path(path).parent) == ".":
        root, ext = os.path.splitext(path)
        if ext != "":
            return True
    return False


def is_bmp_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is `.bmp`, then return True.
    Otherwise, return False.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() == ImageFormat.BMP.value:
        return True
    return False


def is_ckpt_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is `.ckpt`, then return True.
    Otherwise, return False.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["ckpt"]:
        return True
    return False


def is_image_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is in the list of image formats,
    then return True.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ImageFormat.values():
        return True
    return False


def is_json_file(path: str | None) -> bool:
    """
    If the path is a file and the file extension is json, return True.
    Otherwise, return False.
    
    Args:
        path (str | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and path.split(".")[1].lower() in ["json"]:
        return True
    return False


def is_name(path: Path | None) -> bool:
    """
    If the path is None, return False. If the path is the same as the path's
    stem, return True. Otherwise, return False.

    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if path == str(Path(path).stem):
        return True
    return False


def is_stem(path: Path | None) -> bool:
    """
    If the path is not None, and the parent of the path is the current
    directory, and the path has no extension, then the path is a stem.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if str(Path(path).parent) == ".":
        root, ext = path.split(".")
        if ext == "":
            return True
    return False


def is_torch_saved_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is one of the following:
        - pt
        - pth
        - weights
        - ckpt
    Then return True. Otherwise, return False.
    
    Args:
      path (Path | None): The path to the file to be checked.
    
    Returns:
      A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["pt", "pth", "weights", "ckpt"]:
        return True
    return False


def is_txt_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is txt, return True, otherwise
    return False.

    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["txt"]:
        return True
    return False


def is_url(path: Path | None) -> bool:
    """
    If the path is a URL, return True, otherwise return False.
    
    Args:
        path (Path | None): The path to the file or directory.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if isinstance(validators.url(path), validators.ValidationFailure):
        return False
    return True


def is_url_or_file(path: Path | None) -> bool:
    """
    If the path is a URL or a file, return True. Otherwise, return False
    
    Args:
        path (Path | None): The path to the file or URL.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if isinstance(validators.url(path), validators.ValidationFailure) or \
       os.path.isfile(path=path):
        return False
    return True


def is_video_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is in the list of video
    formats, then return True.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in VideoFormat.values():
        return True
    return False


def is_video_stream(path: Path | None) -> bool:
    """
    If the path is not None and contains the string 'rtsp', return True,
    otherwise return False.
    
    Args:
        path (Path | None): The path to the video file or stream.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    return "rtsp" in path.lower()


def is_weights_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is `pt` or `pth`, then return
    True. Otherwise, return False.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["pt", "pth"]:
        return True
    return False


def is_xml_file(path: Path | None) -> bool:
    """
    If the path is a file and the file extension is xml, return True, otherwise
    return False.
    
    Args:
        path (Path | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["xml"]:
        return True
    return False


def is_yaml_file(path: Path | None) -> bool:
    """Check if the given path is a `.yaml` file."""
    if path is None:
        return False
    if os.path.isfile(path=path) and \
       path.split(".")[1].lower() in ["yaml", "yml"]:
        return True
    return False


def list_files(patterns: Strs) -> list[str]:
    """
    It takes a list of file patterns and returns a list of all files that match
    those patterns.
    
    Args:
      patterns (Strs): A list of file paths to search for images.
    
    Returns:
        A list of unique file paths.
    """
    patterns    = to_list(patterns)
    patterns    = [p for p in patterns if p is not None]
    patterns    = unique(patterns)
    image_paths = []
    for pattern in patterns:
        for abs_path in glob(pattern):
            if os.path.isfile(abs_path):
                image_paths.append(abs_path)
    return unique(image_paths)


def list_subdirs(path: Path | None) -> list[str] | None:
    """
    It returns a list of all the subdirectories of the given path
    
    Args:
        path (Path | None): The given path.
    
    Returns:
        A list of all the subdirectories in the given path.
    """
    if path is None:
        return None
    path = str(Path)
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
