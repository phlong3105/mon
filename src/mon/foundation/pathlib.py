#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python :mod:`pathlib` module."""

from __future__ import annotations

__all__ = [
    "Path", "PosixPath", "PurePath", "PurePosixPath", "PureWindowsPath",
    "WindowsPath", "copy_file", "delete_cache", "delete_files", "get_files",
    "get_next_version", "hash_files", "mkdirs", "rmdirs",
]

import glob
import os
import pathlib
import shutil
from pathlib import *

import validators

from mon.foundation import builtins


# region Path

class Path(type(pathlib.Path())):
    """An extension of :class:`pathlib.Path` with more functionalities.
    
    See Also: :class:`pathlib.Path`.
    
    Notes:
        Most of the function here should be properties, but we keep them as
        methods to be consistent with :class:`pathlib.Path`.
    """
    
    def is_basename(self) -> bool:
        """Return True if the current path is a basename of a file. Otherwise,
        return False.
        """
        return str(self) == self.name
    
    def is_bmp_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a bitmap file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".bmp"]
    
    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a checkpoint file. Otherwise,
        return False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".ckpt"]
    
    def is_dir_like(self) -> bool:
        """Return True if the path is a correct file format. """
        return "" in self.suffix
    
    def is_file_like(self) -> bool:
        """Return True if the path is a correct file format. """
        return "." in self.suffix
    
    def is_image_file(self, exist: bool = True) -> bool:
        """Return True if the current path is an image file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) \
            and self.suffix.lower() in [
                ".arw", ".bmp", ".dng", ".jpg", ".jpeg", ".png", ".ppm", ".raf",
                ".tif", ".tiff",
            ]
    
    def is_json_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a JSON file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".json"]
    
    def is_name(self) -> bool:
        """Return True if the current path is the same as the stem. Otherwise,
        return False.
        """
        return self == self.stem
    
    def is_stem(self) -> bool:
        """Return True if the current path isn't None, and the parent of the
        path is the current directory, and the path has no extension. Otherwise,
        return False.
        """
        return str(self) == self.stem
    
    def is_torch_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a file, and the file extension is
        one of the following:
            - pt
            - pt.tar
            - pth
            - pth.tar
            - weights
            - ckpt
            Otherwise, return False.
        """
        return (self.is_file() if exist else True) \
            and self.suffix.lower() in [
                ".pt", ".pth", ".weights", ".ckpt", ".tar"
            ]
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a text file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".txt"]
    
    def is_url(self) -> bool:
        """Return True if the current path is a valid URL. Otherwise, return
        False.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationFailure)
    
    def is_url_or_file(self, exist: bool = True) -> bool:
        """Return True if the path is a file or a valid URL. Otherwise,
        return False.
        """
        return (self.is_file() if exist else True) or \
            not isinstance(validators.url(self), validators.ValidationFailure)
    
    def is_video_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a video file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) \
            and self.suffix.lower() in [
                ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv",
            ]
    
    def is_video_stream(self) -> bool:
        """Return True if the current path is a video stream. Otherwise, return
        False.
        """
        return "rtsp" in str(self).lower()
    
    def is_weights_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a PyTorch's weight file.
        Otherwise, return False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".pt", ".pth"]
    
    def is_xml_file(self, exist: bool = True) -> bool:
        """Return True if the current path is an XML file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".xml"]
    
    def is_yaml_file(self, exist: bool = True) -> bool:
        """Return True if the current path is a YAML file. Otherwise, return
        False.
        """
        return (self.is_file() if exist else True) and self.suffix.lower() in [".yaml", ".yml"]
    
    def has_subdir(self, name: str) -> bool:
        """Return True if a directory has a subdirectory with the given name."""
        subdirs = [d.name for d in self.subdirs()]
        return name in subdirs
    
    def subdirs(self) -> list[Path]:
        """Returns a list of subdirectories' paths inside the current directory.
        """
        path  = self.parent if self.is_file_like() else self
        paths = list(path.iterdir())
        paths = [p for p in paths if p.is_dir()]
        return paths
    
    def files(self) -> list[Path]:
        """Return a list of files' paths inside the current directory."""
        path  = self.parent if self.is_file_like() else self
        paths = list(path.iterdir())
        paths = [p for p in paths if p.is_file()]
        return paths
    
    def latest_file(self) -> Path | None:
        """Return the latest file, in other words, the file with the latest
        created time, in a directory.
        """
        # If the current path is a file, then look for other files inside the
        # same directory.
        files = self.files()
        if len(files) > 0:
            return max(files, key=os.path.getctime)
        return None
    
    def copy_to(self, dst: Path | str, replace: bool = True):
        """Copy a file to a new location.
        
        Args:
            dst: The destination path.
            replace: If True replace the existing file at the destination
                location. Defaults to True.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError()
        mkdirs(dst, parents=True, exist_ok=True)
        dst = dst / self.name if dst.is_dir_like() else dst
        if replace:
            dst.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(dst))


# endregion


# region Obtainment

def get_files(regex: str, recursive: bool = False) -> list[Path]:
    """Get all files matching the given regular expression.
    
    Args:
        regex: A file path patterns.
        recursive: If True, look for file in subdirectories. Defaults to False.
        
    Returns:
        A list of unique file paths.
    """
    paths = []
    for path in glob.glob(regex, recursive=recursive):
        path = Path(path)
        if path.is_file():
            paths.append(path)
    return builtins.unique(paths)


def get_next_version(path: Path | str, prefix: str | None = None) -> int:
    """Get the next version number of items in a directory.
    
    Args:
        path: The directory path containing files.
        prefix: The file prefix. Defaults to None.
    """
    path  = Path(path)
    files = list(path.iterdir())
    existing_versions = []
    for f in files:
        name = f.stem
        if name.startswith("version-") \
            or name.startswith("exp-") \
            or (isinstance(prefix, str) and name.startswith(prefix + "-")):
            ver = name.split("-")[-1].replace("/", "")
            existing_versions.append(int(ver))
    
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1


def hash_files(paths: list[Path | str]) -> int:
    """Return the total hash value of all the files (if it has one). Hash values
    are integers (in bytes) of all files.
    """
    paths = builtins.to_list(paths)
    paths = [Path(f) for f in paths if f is not None]
    return sum(f.stat().st_size for f in paths if f.is_file())


# endregion


# region Creation

def copy_file(src: Path | str, dst: Path | str):
    """Copy a file to a new location.
    
    Args:
        src: The path to the original file.
        dst: The destination path.
    """
    shutil.copyfile(src=str(src), dst=str(dst))


# endregion


# region Alternation

def delete_cache(path: Path | str, recursive: bool = True):
    r"""Clears cache files in a directory and subdirectories.
    
    Args:
        path: The directory path containing the cache files.
        recursive: If True, recursively look for cache files in subdirectories.
            Defaults to True.
    """
    delete_files(regex=".cache", path=path, recursive=recursive)


def delete_files(
    regex    : str,
    path     : Path | str = "",
    recursive: bool       = False
):
    """Delete all files matching the given regular expression.
    
    Args:
        regex: A file path patterns.
        path: A path to a directory to search for the files to delete.
        recursive: If True, look for file in subdirectories. Defaults to False.
    """
    path  = Path(path)
    files = []
    if recursive:
        files += list(path.rglob(*{regex}))
    else:
        files += list(path.glob(regex))
    try:
        for f in files:
            f.unlink()
    except Exception as err:
        print(f"Cannot delete files: {err}.")


def mkdirs(
    paths   : list[Path | str],
    mode    : int  = 0o777,
    parents : bool = True,
    exist_ok: bool = True,
    replace : bool = False,
):
    """Create a new directory at this given path. If mode is given, it is
    combined with the process' umask value to determine the file mode and access
    flags. If the path already exists, FileExistsError is raised.
    
    Args:
        paths: A list of directories' absolute paths.
        mode: If given, it is combined with the process' umask value to
            determine the file mode and access flags.
        parents:
            - If True (the default), any missing parents of this path are
              created as needed; they're created with the default permissions
              without taking mode into account (mimicking the POSIX mkdir -p
              command).
            - If False, a missing parent raises FileNotFoundError.
        exist_ok:
            - If True (the default), FileExistsError exceptions will be ignored
              (same behavior as the POSIX mkdir -p command), but only if the
              last path component isn't an existing non-directory file.
            - If False, FileExistsError is raised if the target directory
              already exists.
        replace: If True, delete existing directories and recreate. Defaults to
            False.
    """
    paths = builtins.to_list(paths)
    paths = builtins.unique(paths)
    try:
        for p in paths:
            p = Path(p)
            if p.is_url():
                continue
            p = p.parent if p.is_file_like() else p
            if replace:
                delete_files(regex="*", path=p)
                p.rmdir()
            p.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
    except Exception as err:
        print(f"Cannot create directory: {err}.")


def rmdirs(paths: list[pathlib.Path | str]):
    """Delete directories.
    
    Args:
        paths: A list of directories' absolute paths.
    """
    paths = builtins.to_list(paths)
    paths = builtins.unique(paths)
    try:
        for p in paths:
            p = Path(p)
            if p.is_url():
                continue
            p = p.parent if p.is_file_like() else p
            delete_files(regex="*", path=p)
            p.rmdir()
    except Exception as err:
        print(f"Cannot delete directory: {err}.")

# endregion
