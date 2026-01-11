#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extended pathlib utilities and domain-specific Path subclass.

This module provides a Path subclass and helpers for file-type checks,
related-file resolution, simple filesystem operations, and download and delete
helpers, wrapping domain-specific behavior on top of pathlib.Path for project
use.
"""

from __future__ import annotations

__all__ = [
    "Path",
    "PosixPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "WindowsPath",
    "delete_files",
    "download_url_to_file",
]

import shutil
from pathlib import (
    Path as Path_,
    PosixPath,
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    WindowsPath,
)
from typing import Iterator, Optional

import requests
import validators

from mon.core.console import console, error_console
from mon.core.enum import (
    ConfigExtension,
    ImageExtension,
    VideoExtension,
    WeightExtension,
)
from mon.core.utils import snakecase


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class Path(type(Path_())):
    """A Path subclass with domain-specific helpers.

    This class extends the standard `pathlib.Path` with methods for file-type
    validation, related-file resolution, and enhanced filesystem operations
    tailored for computer vision and machine learning workflows.
    """
    
    def hash(self) -> int:
        """Return a simple hash based on the file's size."""
        return self.stat().st_size if self.is_file() else 0
    
    # --- Validation ---
    def is_basename(self) -> bool:
        """Return True if the path contains no directory parts (e.g., 'file.txt')."""
        return str(self) == self.name

    def is_stem(self) -> bool:
        """Return True if the path consists only of a stem, with no directory or suffix."""
        return str(self) == self.stem

    def is_url(self) -> bool:
        """Return True if the path string is a valid URL."""
        return not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Return True if the path is a TXT file."""
        return self._check_suffix({".txt"}, exist=exist)

    def is_json_file(self, exist: bool = True) -> bool:
        """Return True if the path is a JSON file."""
        return self._check_suffix({".json"}, exist=exist)

    def is_xml_file(self, exist: bool = True) -> bool:
        """Return True if the path is an XML file."""
        return self._check_suffix({".xml"}, exist=exist)

    def is_yaml_file(self, exist: bool = True) -> bool:
        """Return True if the path is a YAML or YML file."""
        return self._check_suffix({".yaml", ".yml"}, exist=exist)

    def is_image_file(self, exist: bool = True) -> bool:
        """Return True if the path has a recognized image extension."""
        return self._check_suffix(set(ImageExtension.values()), exist=exist)

    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Return True if the path is a raw image format (e.g., DNG, ARW)."""
        return self._check_suffix({".dng", ".arw"}, exist=exist)

    def is_video_file(self, exist: bool = True) -> bool:
        """Return True if the path is a recognized video file."""
        return self._check_suffix(set(VideoExtension.values()), exist=exist)

    def is_video_stream(self) -> bool:
        """Return True if the path appears to be a video stream URL (e.g., RTSP)."""
        return "rtsp" in str(self).lower()

    def is_cache_file(self, exist: bool = True) -> bool:
        """Return True if the path has a .cache suffix."""
        return self._check_suffix({".cache"}, exist=exist)

    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Return True if the path has a .ckpt suffix."""
        return self._check_suffix({".ckpt"}, exist=exist)

    def is_config_file(self, exist: bool = True) -> bool:
        """Return True if the path matches known config extensions."""
        return self._check_suffix(set(ConfigExtension.values()), exist=exist)

    def is_onnx_file(self, exist: bool = True) -> bool:
        """Return True if the path has an .onnx suffix."""
        return self._check_suffix({".onnx"}, exist=exist)

    def is_py_file(self, exist: bool = True) -> bool:
        """Return True if the path has a .py suffix."""
        return self._check_suffix({".py"}, exist=exist)

    def is_weights_file(self, exist: bool = True) -> bool:
        """Return True if the path matches known weight file extensions."""
        return self._check_suffix(set(WeightExtension.values()), exist=exist)
    
    def _check_suffix(self, suffixes: set, exist: bool = True) -> bool:
        """Check if the path has a suffix from the given set and optionally exists.

        Args:
            suffixes: A set of lower-case suffixes to check against (e.g., {".jpg", ".png"}).
            exist: If True, require that the file exists on the filesystem.

        Returns:
            True if the path's suffix is in the set and it meets the existence check.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in suffixes

    # --- Resolve ---
    def normalize(self, exist: bool = False, mkdir: bool = False) -> "Path":
        """Standardize the path by expanding user tags and resolving symlinks.

        Args:
            exist: If True, raises FileNotFoundError if the path doesn't exist.
            mkdir: If True, creates the necessary parent directories.

        Returns:
            A resolved and absolute Path object.

        Raises:
            FileNotFoundError: If `exist` is True and the path does not exist.
        """
        path = self.expanduser().resolve()
        
        if exist and not path.exists():
            raise FileNotFoundError(f"Path not found at: {path}")
            
        if mkdir:
            # If path is file-like, create parent; otherwise, create the dir itself.
            dir_to_create = path.parent if path.is_file() else path
            dir_to_create.mkdir(parents=True, exist_ok=True)
            
        return path
    
    def has_subdir(self, name: str) -> bool:
        """Return True if the directory contains a subdirectory with the given name.

        Args:
            name: The subdirectory name to look for.
        """
        return name in [d.name for d in self.subdirs()]
    
    def subdirs(self, recursive: bool = False) -> Iterator["Path"]:
        """Return an iterator over subdirectories of this path.

        Args:
            recursive: If True, include nested subdirectories.

        Returns:
            An iterator over subdirectory Path objects.
        """
        path        = self.parent if self.is_file() else self
        search_root = path.rglob("*") if recursive else path.iterdir()
        return (p for p in search_root if p.is_dir())

    def files(self, recursive: bool = False) -> list["Path"]:
        """Return a list of files under this path.

        Args:
            recursive: If True, include files in nested directories.

        Returns:
            A list of file Path objects.
        """
        path  = self.parent if self.is_file() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]

    def txt_file(self) -> "Path":
        """Return a matching .txt file if present."""
        txt_path = self.with_suffix(".txt")
        return txt_path if txt_path.is_file() else self
    
    def json_file(self) -> "Path":
        """Return a matching .json file if present."""
        json_path = self.with_suffix(".json")
        return json_path if json_path.is_file() else self
    
    def xml_file(self) -> "Path":
        """Return a matching .xml file if present."""
        xml_path = self.with_suffix(".xml")
        return xml_path if xml_path.is_file() else self
    
    def yaml_file(self) -> "Path":
        """Return a matching YAML file (.yaml, .yml) if present."""
        for ext in [".yaml", ".yml"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self
    
    def image_file(self) -> "Path":
        """Return a matching image file based on known extensions."""
        for ext in ImageExtension.values():
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self
    
    def video_file(self) -> "Path":
        """Return a matching video file based on known extensions."""
        for ext in VideoExtension.values():
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self
    
    def cache_file(self) -> "Path":
        """Return a matching .cache file if present."""
        cache_path = self.with_suffix(".cache")
        return cache_path if cache_path.is_file() else self
    
    def ckpt_file(self) -> "Path":
        """Return a .ckpt Path in the same directory if it exists."""
        ckpt_path = self.with_suffix(".ckpt")
        return ckpt_path if ckpt_path.is_file() else self

    def config_file(self) -> "Path":
        """Return the first matching configuration file in the same directory."""
        stems = {self.stem, snakecase(self.stem)}
        for f in self.parent.iterdir():
            if f.stem in stems and f.suffix.lower() in ConfigExtension.values():
                return f
        return self
    
    def onnx_file(self) -> "Path":
        """Return a matching .onnx file if present."""
        onnx_path = self.with_suffix(".onnx")
        return onnx_path if onnx_path.is_file() else self
    
    def py_file(self) -> "Path":
        """Return a matching .py file if present."""
        py_path = self.with_suffix(".py")
        return py_path if py_path.is_file() else self
    
    def label_file(self) -> "Path":
        """Return a matching label file (.txt, .xml, .json) if present."""
        for ext in [".txt", ".xml", ".json"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def latest_file(self) -> Optional["Path"]:
        """Return the most recently modified file in the directory."""
        files = self.files()
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    
    def relative_path(self, start_part: str) -> "Path":
        """Return a new Path starting from the first occurrence of `start_part`.

        Args:
            start_part: The substring to start the new relative path from.

        Returns:
            A Path starting at the first occurrence of `start_part`, or the
            original Path if `start_part` is not found.
        """
        path_str   = str(self)
        start_part = str(start_part)
        if start_part not in path_str:
            return self
        start_idx = path_str.find(start_part)
        return Path(path_str[start_idx:])
    
    # --- Modification ---
    def replace_part(self, old: str, new: str, count: int = 1) -> "Path":
        """Return a new Path with part of the string replaced.

        Args:
            old: The substring to replace.
            new: The replacement substring.
            count: The maximum number of replacements to perform.
        """
        return Path(str(self).replace(old, new, count))
    
    # --- Copying ---
    def copy_to(self, dst: str | "Path", replace: bool = True):
        """Copy the current file to a destination, creating parents as needed.

        Args:
            dst: The destination path or directory.
            replace: If True, remove any existing destination file.

        Raises:
            NotImplementedError: If `dst` is a URL.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError("This method is not yet supported.")
        
        destination = dst / self.name if dst.is_dir() else dst
        destination.parent.mkdir(parents=True, exist_ok=True)
        if replace:
            destination.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(destination))
    
    # --- Deletion ---
    def rmdir(self, recursive: bool = True):
        """Remove the directory, using `shutil.rmtree` for recursive deletion."""
        if recursive and self.is_dir():
            shutil.rmtree(self)
        elif self.is_dir():
            super().rmdir()

# endregion


# ==============================================================================
# region FILESYSTEM
# ==============================================================================

def delete_files(path: str | Path, regex: str = None, recursive: bool = False):
    """Delete files matching a pattern under a given path.

    Args:
        path: The path or directory to delete from.
        regex: A glob pattern to match files (e.g., "*.jpg"). Defaults to None.
        recursive: If True, search recursively for matches.
    """
    path = Path(path)
    
    if not regex:
        # If no pattern, delete the single path if it's a file.
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                # Safety check: Do not delete directories without a pattern.
                console.log(f"Path is a directory. To delete, use `path.rmdir()`.")
        except Exception as err:
            error_console.log(f"Could not delete {path}: {err}")
        return
    
    # If a pattern is given, search for matching files and delete them.
    search_root     = path if path.is_dir() else path.parent
    files_to_delete = search_root.rglob(regex) if recursive else search_root.glob(regex)
    
    for f in files_to_delete:
        try:
            if f.is_file():
                f.unlink()
        except Exception as err:
            error_console.error(f"Failed to delete {f}: {err}")


def download_url_to_file(url: str, path: str | Path, overwrite: bool = False) -> Path:
    """Download a file from a URL to the local filesystem with a progress bar.

    Args:
        url: The source URL to download from.
        path: The destination path to save the file.
        overwrite: If True, overwrite the destination file if it exists.

    Returns:
        The destination Path object.

    Raises:
        ValueError: If `url` is not a valid URL.
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
