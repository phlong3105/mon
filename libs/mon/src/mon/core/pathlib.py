#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extended pathlib utilities and domain-specific Path subclass.

This module provides a Path subclass and helpers for file-type checks,
related-file resolution, simple filesystem operations, and download and delete
helpers, wrapping domain-specific behavior on top of pathlib.Path for project
use.
"""

__all__ = [
    "Path",
    "PosixPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "WindowsPath",
    "download_url_to_file",
]

import os
import shutil
from pathlib import (
    Path as Path_,
    PosixPath,
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    WindowsPath,
)

import validators

from mon.core.enum import (
    ConfigExtension,
    ImageExtension,
    VideoExtension,
    WeightExtension,
)
from mon.core.utils import snakecase


# ==============================================================================
# PATH ENGINE
# ==============================================================================

class Path(type(Path_())):
    """A Path subclass with domain-specific helpers.

    Provide file-type checks, listing helpers, related-file resolution, copy
    and replace operations, and deletion utilities on top of pathlib.Path.
    """

    # --- Properties ---
    def hash(self) -> int:
        """Return a simple hash based on file size.

        Return the file size in bytes when the path is a file, otherwise 0.
        """
        return self.stat().st_size if self.is_file() else 0

    # --- Validation (Base) ---
    def is_basename(self) -> bool:
        """Return True if the path equals its basename.

        Return True when the path is a bare filename with no directory
        component.
        """
        return str(self) == self.name

    def is_name(self) -> bool:
        """Return True if the path equals its name.

        Return True when the path string equals its stem or filename.
        """
        return str(self) == self.stem

    def is_stem(self) -> bool:
        """Return True if the path equals its stem.

        Return True when the path string equals its stem (filename without
        suffix).
        """
        return str(self) == self.stem

    def is_url(self) -> bool:
        """Return True if the path is a URL.

        Return True when validators.url recognizes the path as a URL.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationError)

    def is_url_or_file(self, exist: bool = True) -> bool:
        """Return True if path is an existing file or a valid URL.

        Args:
            exist: If True, require local files to exist.

        Return True if the path is an existing file (when requested) or a
        valid URL.
        """
        return (
            (not exist or self.is_file())
            or not isinstance(validators.url(str(self)), validators.ValidationError)
        )

    def is_file_like(self) -> bool:
        """Return True if the path appears to represent a file.

        Return True when the path has a suffix suggesting a file.
        """
        return "." in self.suffix

    def is_dir_like(self) -> bool:
        """Return True if the path appears to represent a directory.

        Return True when the path has no suffix.
        """
        return self.suffix == ""

    def has_subdir(self, name: str) -> bool:
        """Return True if the directory contains a named subdirectory.

        Args:
            name: Subdirectory name to look for.

        Return True when a subdirectory with the given name exists.
        """
        return name in [d.name for d in self.subdirs()]

    # --- Validation (Text) ---
    def is_json_file(self, exist: bool = True) -> bool:
        """Return True if the path is a JSON file.

        Args:
            exist: If True, require that the file exists.

        Return True when the suffix is .json (and the file exists if
        requested).
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".json"

    def is_txt_file(self, exist: bool = True) -> bool:
        """Return True if the path is a TXT file.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a .txt suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".txt"

    def is_xml_file(self, exist: bool = True) -> bool:
        """Return True if the path is an XML file.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a .xml suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".xml"

    def is_yaml_file(self, exist: bool = True) -> bool:
        """Return True if the path is a YAML or YML file.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a .yaml or .yml suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".yaml", ".yml"]

    # --- Validation (Image File) ---
    def is_image_file(self, exist: bool = True) -> bool:
        """Return True if the path has a recognized image extension.

        Args:
            exist: If True, require the file to exist.

        Return True when the suffix matches known image extensions.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in ImageExtension

    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Return True if the path is a raw image format.

        Args:
            exist: If True, require that the file exists.

        Return True when the suffix matches raw image extensions.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".dng", ".arw"]

    # --- Validation (Video File) ---
    def is_video_file(self, exist: bool = True) -> bool:
        """Return True if the path is a recognized video file.

        Args:
            exist: If True, require that the file exists.

        Return True when the suffix matches known video extensions.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in VideoExtension

    def is_video_stream(self) -> bool:
        """Return True if the path appears to be a video stream URL.

        Check for common stream prefixes such as rtsp.
        """
        return "rtsp" in str(self).lower()

    # --- Validation (ML File) ---
    def is_cache_file(self, exist: bool = True) -> bool:
        """Return True if the path has a .cache suffix.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a .cache suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".cache"

    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Return True if the path has a .ckpt suffix.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a .ckpt suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".ckpt"

    def is_config_file(self, exist: bool = True) -> bool:
        """Return True if the path matches known config extensions.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a known config file extension.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in ConfigExtension

    def is_onnx_file(self, exist: bool = True) -> bool:
        """Return True if the path has an .onnx suffix.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has an .onnx suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".onnx"

    def is_py_file(self, exist: bool = True) -> bool:
        """Return True if the path has a .py suffix.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a .py suffix.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".py"

    def is_weights_file(self, exist: bool = True) -> bool:
        """Return True if the path matches weight file extensions.

        Args:
            exist: If True, require that the file exists.

        Return True when the file has a weight file extension.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in WeightExtension

    # --- Relationship Resolvers ---
    def subdirs(self, recursive: bool = False) -> list["Path"]:
        """Return subdirectories of this path.

        Args:
            recursive: If True, include nested subdirectories.

        Returns:
            List of subdirectory Path objects.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_dir()]

    def files(self, recursive: bool = False) -> list["Path"]:
        """Return files under this path.

        Args:
            recursive: If True, include files in nested directories.

        Returns:
            List of file Path objects.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]

    def ckpt_file(self) -> "Path":
        """Return a .ckpt Path if present.

        Return the resolved checkpoint file or self if not found.
        """
        ckpt_path = self.with_suffix(".ckpt")
        return ckpt_path if ckpt_path.is_file() else self

    def config_file(self) -> "Path":
        """Return the first matching configuration file.

        Try common config extensions and snakecased stems when searching.
        """
        for ext in ConfigExtension.values():
            for stem in [self.stem, snakecase(self.stem)]:
                config_path = self.with_name(f"{stem}{ext}")
                if config_path.is_file():
                    return config_path
        return self

    def label_file(self) -> "Path":
        """Return a matching label file if present.

        Return the located label file or self when none found.
        """
        for ext in [".txt", ".xml", ".json"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def latest_file(self) -> "Path":
        """Return the most recently created file in the directory.

        Return the newest file by creation time, or None if no files.
        """
        files = self.files()
        return max(files, key=os.path.getctime) if files else None

    def image_file(self) -> "Path":
        """Return a matching image file according to known extensions.

        Return the located image file or self when none found.
        """
        for ext in ImageExtension.values():
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def txt_file(self) -> "Path":
        """Return a matching .txt file if present.

        Return the located .txt file or self when none found.
        """
        for ext in [".txt"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def yaml_file(self) -> "Path":
        """Return a matching YAML file if present.

        Return the located YAML file or self when none found.
        """
        for ext in [".yaml", ".yml"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def relative_path(self, start_part: str) -> "Path":
        """Return a new Path starting from the first occurrence of start_part.

        Args:
            start_part: Substring to start the returned relative path.

        Returns:
            A Path that begins at the first occurrence of start_part in the
            original path, or the original Path if start_part is not found.
        """
        path       = Path(self)
        start_part = str(start_part)
        path_str   = str(path)
        if start_part not in path_str:
            return path
        start_idx = path_str.index(start_part)
        return Path(path_str[start_idx:])

    # --- Creation ---
    def copy_to(self, dst: str, replace: bool = True):
        """Copy the current file to dst, creating parents as needed.

        Args:
            dst: Destination path or directory.
            replace: If True, remove any existing destination file.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError("``dst`` as a URL is not supported.")
        dst = dst / self.name if dst.is_dir_like() else dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if replace:
            dst.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(dst))

    def replace_part(self, old: str, new: str, count: int = 1) -> "Path":
        """Return a new Path with part of the string replaced.

        Args:
            old: Substring to replace.
            new: Replacement substring.
            count: Maximum number of replacements to perform.
        """
        return Path(str(self).replace(old, new, count))

    # --- Deletion ---
    def rmdir(self):
        """Remove the directory and its contents.

        Delete all files under the path and then remove the directory.
        """
        delete_files(path=self, regex="*", recursive=True)
        super().rmdir()


# ==============================================================================
# REMOTE & EXTERNAL IO
# ==============================================================================

# --- Ingestion ---
def download_url_to_file(url: str, path: Path, overwrite: bool = False) -> Path:
    """Download a file from a URL to the local filesystem.

    Args:
        url: Source URL.
        path: Destination path.
        overwrite: If True, overwrite an existing destination file.

    Returns:
        The destination Path where the file was saved.

    Raises:
        ValueError: If ``url`` is not a valid URL.
    """
    if not Path(url).is_url():
        raise ValueError(f"url must be a valid URL, got {url}.")

    path = Path(path)
    if not path.exists() or overwrite:
        path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.hub.download_url_to_file(url, str(path), None, True)
    return path


# ==============================================================================
# FILESYSTEM ORCHESTRATION
# ==============================================================================

# --- Cleanup Operations ---
def delete_files(path: Path, regex: str = None, recursive: bool = False):
    """Delete files matching a pattern under the given path.

    Args:
        path: Path or directory to delete from.
        regex: Glob pattern to match files. Defaults to None.
        recursive: If True, search recursively for matches.

    Raises:
        Exception: If a file cannot be deleted, the exception is printed.
    """
    path = Path(path)
    if regex:
        path  = path.parent if not path.is_dir() else path
        files = list(path.rglob(regex)) if recursive else list(path.glob(regex))
    else:
        files = [path]
    for f in files:
        try:
            f.unlink()
        except Exception as err:
            print(f"Cannot delete file: {err}.")
