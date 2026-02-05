#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extended pathlib utilities and domain-specific Path subclass.

This module provides a Path subclass with additional methods for file-type
validation, related-file resolution, and enhanced filesystem operations.
"""

from __future__ import annotations

__all__ = [
    "Path",
    "PosixPath",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "WindowsPath",
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
from typing import Optional

import validators

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
    """Path subclass with domain-specific helpers.

    Extend the standard pathlib.Path with methods for file-type validation,
    related-file resolution, and enhanced filesystem operations.
    """

    def hash(self) -> int:
        """Return a simple hash based on the file's size."""
        return self.stat().st_size if self.is_file() else 0

    # --- Validation ---
    def is_basename(self) -> bool:
        """Check if the path contains no directory parts."""
        return str(self) == self.name

    def is_stem(self) -> bool:
        """Check if the path consists only of a stem."""
        return str(self) == self.stem

    def is_url(self) -> bool:
        """Check if the path string is a valid URL."""
        return not isinstance(validators.url(str(self)), validators.ValidationError)

    def is_txt_file(self, exist: bool = True) -> bool:
        """Check if the path is a TXT file.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".txt"}, exist=exist)

    def is_json_file(self, exist: bool = True) -> bool:
        """Check if the path is a JSON file.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".json"}, exist=exist)

    def is_xml_file(self, exist: bool = True) -> bool:
        """Check if the path is an XML file.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".xml"}, exist=exist)

    def is_yaml_file(self, exist: bool = True) -> bool:
        """Check if the path is a YAML or YML file.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".yaml", ".yml"}, exist=exist)

    def is_image_file(self, exist: bool = True) -> bool:
        """Check if the path has a recognized image extension.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix(set(ImageExtension.values()), exist=exist)

    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Check if the path is a raw image format.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".dng", ".arw"}, exist=exist)

    def is_video_file(self, exist: bool = True) -> bool:
        """Check if the path is a recognized video file.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix(set(VideoExtension.values()), exist=exist)

    def is_video_stream(self) -> bool:
        """Check if the path appears to be a video stream URL."""
        return "rtsp" in str(self).lower()

    def is_cache_file(self, exist: bool = True) -> bool:
        """Check if the path has a .cache suffix.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".cache"}, exist=exist)

    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Check if the path has a .ckpt suffix.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".ckpt"}, exist=exist)

    def is_config_file(self, exist: bool = True) -> bool:
        """Check if the path matches known config extensions.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix(set(ConfigExtension.values()), exist=exist)

    def is_onnx_file(self, exist: bool = True) -> bool:
        """Check if the path has an .onnx suffix.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".onnx"}, exist=exist)

    def is_py_file(self, exist: bool = True) -> bool:
        """Check if the path has a .py suffix.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix({".py"}, exist=exist)

    def is_weights_file(self, exist: bool = True) -> bool:
        """Check if the path matches known weight file extensions.

        Args:
            exist: If True, require that the file exists. Defaults to True.
        """
        return self._check_suffix(set(WeightExtension.values()), exist=exist)

    def has_subdir(self, name: str) -> bool:
        """Check if the directory contains a subdirectory with the given name.

        Args:
            name: Subdirectory name to look for.
        """
        return name in [d.name for d in self.subdirs()]

    def _check_suffix(self, suffixes: set, exist: bool = True) -> bool:
        """Check if the path has a suffix from the given set.

        Args:
            suffixes: Set of lower-case suffixes to check against.
            exist: If True, require that the file exists. Defaults to True.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in suffixes

    # --- Discovery ---
    def normalize(self, exist: bool = False, mkdir: bool = False) -> "Path":
        """Standardize the path by expanding user tags and resolving symlinks.

        Args:
            exist: If True, raises FileNotFoundError if the path doesn't exist.
                Defaults to False.
            mkdir: If True, creates the necessary parent directories.
                Defaults to False.

        Raises:
            FileNotFoundError: If ``exist`` is True and the path does not exist.
        """
        path = self.expanduser().resolve()

        if exist and not path.exists():
            raise FileNotFoundError(f"Path not found at: {path}")

        if mkdir:
            # If path is file-like, create parent; otherwise, create the dir itself.
            dir_to_create = path if path.is_dir() else path.parent
            dir_to_create.mkdir(parents=True, exist_ok=True)

        return path

    def subdirs(self, recursive: bool = False) -> list["Path"]:
        """Return an iterator over subdirectories of this path.

        Args:
            recursive: If True, include nested subdirectories. Defaults to False.
        """
        root         = self.parent if self.is_file() else self
        search_paths = root.rglob("*") if recursive else root.iterdir()
        return [p for p in search_paths if p.is_dir()]

    def data_dir(self) -> Optional["Path"]:
        """Go up the directory tree to find a 'data' directory."""
        current_dir = self.parent if self.is_file() else self
        for parent in current_dir.parents:
            if (parent / "data").exists():
                return parent / "data"
        return None

    def image_dirs(self, recursive: bool = False) -> list["Path"]:
        """Return an iterator over image directories under this path.

        Args:
            recursive: If True, include nested subdirectories. Defaults to False.
        """
        root         = self.parent if self.is_file() else self
        search_paths = root.rglob("image") if recursive else root.iterdir()
        return [p for p in search_paths if p.is_dir() and p.stem == "image"]

    def files(self, recursive: bool = False) -> list["Path"]:
        """Return a list of files under this path.

        Args:
            recursive: If True, include files in nested directories.
                Defaults to False.
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
        """Return a matching YAML file if present."""
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
        """Return a matching label file if present."""
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
        """Return a new Path starting from the first occurrence of ``start_part``.

        Args:
            start_part: Substring to start the new relative path from.
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
            old: Substring to replace.
            new: Replacement substring.
            count: Maximum number of replacements to perform. Defaults to 1.
        """
        return Path(str(self).replace(old, new, count))

    # --- Copying ---
    def copy_to(self, dst: str | "Path", replace: bool = True):
        """Copy the current file to a destination.

        Args:
            dst: Destination path or directory.
            replace: If True, remove any existing destination file.
                Defaults to True.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
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
        """Remove the directory.

        Args:
            recursive: If True, use ``shutil.rmtree`` for recursive deletion.
                Defaults to True.
        """
        if recursive and self.is_dir():
            shutil.rmtree(self)
        elif self.is_dir():
            super().rmdir()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
