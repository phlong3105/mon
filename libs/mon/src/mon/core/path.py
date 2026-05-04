#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Path.

This module provides custom Path implementations using pathlib.
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
from typing import Iterable, Literal, Optional, Union

from .dtype import (
    ConfigExtension,
    ImageExtension,
    VideoExtension,
    WeightExtension,
)
from .utils import truncate_string


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Path(type(Path_())):  # Dynamic inheritance based on OS
    """Custom Path."""

    # --- Lifecycle & Initialization ---
    def __init__(self, *args):
        """Override constructor to allow empty path components."""
        if args and args[0] is None:
            args = ("",) + args[1:]
        super().__init__(*args)

    # --- Properties---
    @property
    def size(self) -> int:
        """Return file size in bytes. Returns 0 if file does not exist."""
        return self.stat().st_size if self.is_file() else 0

    @property
    def created_at(self) -> float:
        """Return creation timestamp."""
        return self.stat().st_ctime if self.exists() else 0.0

    @property
    def modified_at(self) -> float:
        """Return modification timestamp."""
        return self.stat().st_mtime if self.exists() else 0.0

    @property
    def hash(self) -> int:
        """Return a simple hash based on the file's size and modification time."""
        return hash((self.size, self.modified_at))

    @property
    def txt_file(self) -> "Path":
        """Return a matching .txt file if present."""
        return self.sibling(".txt")

    @property
    def json_file(self) -> "Path":
        """Return a matching .json file if present."""
        return self.sibling(".json")

    @property
    def xml_file(self) -> "Path":
        """Return a matching .xml file if present."""
        return self.sibling(".xml")

    @property
    def yaml_file(self) -> "Path":
        """Return a matching .yaml file if present."""
        return self.sibling(".yaml")

    @property
    def image_file(self) -> "Path":
        """Return a matching image file if present."""
        return self.find_sibling_ext(ImageExtension.values())

    @property
    def ckpt_file(self) -> "Path":
        """Return a matching .ckpt file if present."""
        return self.sibling(".ckpt")

    @property
    def onnx_file(self) -> "Path":
        """Return a matching .onnx file if present."""
        return self.sibling(".onnx")

    # --- Discovery ---
    def files(self, *patterns, recursive: bool = False) -> list["Path"]:
        """List files in this directory. Safe against non-existent paths.

        Args:
            *patterns: Glob patterns to filter files (e.g. "*.jpg"). If empty,
                returns an empty list.
            recursive (bool, optional): If True, include files in subdirectories.
                Defaults to False.
        """
        # Check if the directory exists first
        root = self._get_search_root()
        if not root.exists():
            return []

        # Loop over patterns
        files = []
        for pattern in patterns:
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            files += [p for p in iterator if p.is_file()]

        return files

    def subdirs(self, recursive: bool = False) -> list["Path"]:
        """List subdirectories.

        Args:
            recursive (bool, optional): If True, include nested subdirectories.
                Defaults to False.
        """
        root = self._get_search_root()
        if not root.exists():
            return []

        iterator = root.rglob("*") if recursive else root.iterdir()
        return [p for p in iterator if p.is_dir()]

    def latest_file(self) -> Optional["Path"]:
        """Return the most recently modified file."""
        files = self.files("*")
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    def find_sibling_ext(self, *extensions: str | Iterable[str]) -> "Path":
        """Checks if a file with the same stem exists with one of the given
        extensions.

        Args:
            *extensions (str | Iterable[str]): List of extensions to check.

        Returns:
            Path: The first matching file path found, or self if None found.
        """
        # Flatten inputs
        exts = set()
        for item in extensions:
            if isinstance(item, (set, list, tuple)):
                exts.update(item)
            else:
                exts.add(item)
        # Normalize to lowercase
        extensions = [e if e.startswith(".") else f".{e}" for e in exts]

        for ext in extensions:
            candidate = self.with_suffix(ext)
            if candidate.is_file():
                return candidate
        return self

    def _get_search_root(self) -> "Path":
        """Internal helper to determine where to start searching."""
        if self.is_file():
            return self.parent
        if self.is_dir():
            return self
        # Fallback for non-existent paths: assume it's meant to be a dir
        return self

    # --- Validation ---
    def has_name(self, name: str, exists: bool = False) -> bool:
        """Check if the path has the given name (either stem or full name).

        Args:
            name (str): Name to check against (e.g. "file" or "file.txt").
            exists (bool, optional): If True, also check if the file exists.
                Defaults to False.
        """
        if exists and not self.exists():
            return False

        return self.name == name or self.stem == name

    def has_ext(self, *extensions: str | Iterable[str], exists: bool = False) -> bool:
        """Robust extension check.

        Args:
            *extensions (str | Iterable[str]): List of extensions to check.
            exists (bool, optional): If True, check if the file exists.
                Defaults to False.
        """
        if exists and not self.is_file():
            return False

        # Flatten inputs
        exts = set()
        for item in extensions:
            if isinstance(item, (set, list, tuple)):
                exts.update(item)
            else:
                exts.add(item)

        # Normalize to lowercase
        exts = {str(e) for e in exts}
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

        return self.suffix.lower() in exts

    def is_image_file(self, exists: bool = True) -> bool:
        """Check if the path has a recognized image extension.

        Args:
            exists (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(ImageExtension.values(), exists=exists)

    def is_raw_image_file(self, exists: bool = True) -> bool:
        """Check if the path is a raw image format.

        Args:
            exists (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext({".dng", ".arw"}, exists=exists)

    def is_video_file(self, exists: bool = True) -> bool:
        """Check if the path is a recognized video file.

        Args:
            exists (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(VideoExtension.values(), exists=exists)

    def is_weights_file(self, exists: bool = True) -> bool:
        """Check if the path matches known weight file extensions.

        Args:
            exists (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(WeightExtension.values(), exists=exists)

    def is_config_file(self, exists: bool = True) -> bool:
        """Check if the path matches known config extensions.

        Args:
            exists (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(ConfigExtension.values(), exists=exists)

    def is_url(self) -> bool:
        """Fast check for URL scheme without external dependencies."""
        return str(self).startswith(("http://", "https://", "rtsp://", "ftp://"))

    # --- Retrieval ---
    def sibling(self, ext: str) -> "Path":
        """Return a path with the same parent and stem but different extension.

        Args:
            ext (str): New extension (with or without dot).
        """
        # Ensure ext has dot
        ext = ext if ext.startswith(".") else f".{ext}"
        return self.with_suffix(ext)

    def resolve_subdir(self, dirname: str) -> "Path":
        """Smartly resolves a subdirectory.

        If the current path already ends with ``dirname``, returns self.
        Otherwise, appends ``dirname`` to self.

        Examples:
            >>> Path("data/coco").resolve_subdir("images")         # -> "data/coco/images"
            >>> Path("data/coco/images").resolve_subdir("images")  # -> "data/coco/images"

        Args:
            dirname (str): Name of the subdirectory to resolve.
        """
        # Normalize to handle trailing slashes or mixed separators
        # We use .name instead of .stem to correctly handle folders with dots (e.g. "v1.0")
        if (self / dirname).is_dir():
            return self / dirname
        return self

    # --- Mutation ---
    def append(self, path: Union["Path", str]):
        """Append a path to the current path, ensuring no duplicate parts."""
        # If path is empty or None, return self
        if not path:
            return self

        # Normalize inputs
        path: Path = Path(path)

        # If path is a file, use its stem as the new path
        if path.is_file():
            path = path.stem if path.suffix else path.name
            path = str(path).strip()
            if path and self.name != path:
                return self / path

        # If the first part of the path is the same as self.name, skip it to
        # avoid duplication
        if path.parts[0] == self.name:
            return self.parent / path
        else:
            return self / path

    # --- Computation ---
    def commonpath_to(self, other: "Path") -> "Path":
        """Return the longest common path prefix between two paths.

        Args:
            other (Path): Another path to compare with.
        """
        return Path(os.path.commonpath([str(self), str(other)]))

    def unique_path_from(self, other: "Path") -> "Path":
        """Return a unique path based on the current path and another path.

        Args:
            other (Path): Another path to compare with.
        """
        commonpath = self.commonpath_to(other)
        if commonpath == self:
            return Path()
        else:
            return self.replace_part(str(commonpath), "")

    def relative_path_to(self, start_part: str) -> "Path":
        """Return a new Path starting from the first occurrence of ``start_part``.

        Example: Path("/a/b/c/d").relative_path_to("b") -> Path("b/c/d")

        Args:
            start_part (str): Substring to start the new relative path from.
        """
        # Normalize inputs
        start_part = str(start_part).strip()

        try:
            # Find the index of the folder in the path parts tuple
            # This ensures we match exact folder names, not substrings
            idx = self.parts.index(start_part)
            return Path(*self.parts[idx:])
        except ValueError:
            return self

    # --- Transformation ---
    def normalize(self) -> "Path":
        """Resolves symlinks, '..', and expands user."""
        return self.expanduser().resolve()

    def truncate(
        self,
        max_length: int = 80,
        side: Literal["left", "middle", "right"] = "middle"
    ) -> str:
        """Return a truncated string representation of the path.

        Args:
            max_length (int, optional): Maximum length of the truncated string.
                Defaults to 80.
            side (Literal["left", "middle", "right"], optional): Which side to
                truncate. Defaults to "middle".
        """
        return truncate_string(str(self), max_length=max_length, side=side)

    def ensure_dir(self, is_file: bool = False) -> "Path":
        """Creates the directory for this path.

        Args:
            is_file (bool, optional): Explicitly mark this path as a file.
                If False (default) it attempts to guess based on suffix,
                but this fails for folders like "v1.0". Defaults to False.
        """
        # If it exists, we know what it is
        if self.exists():
            if self.is_dir():
                return self
            if self.is_file():
                self.parent.mkdir(parents=True, exist_ok=True)
                return self

        # Heuristic for non-existent paths
        if is_file or self.suffix:
            self.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.mkdir(parents=True, exist_ok=True)
        return self

    def replace_part(self, old: str, new: str) -> "Path":
        """Return a new Path with part of the string replaced.

        Args:
            old (str): Substring to replace.
            new (str): Replacement substring.
        """
        tmp = str(self)
        tmp = tmp.replace(str(old), str(new))
        return Path(tmp)

    # --- Filesystem ---
    def copy_to(self, dst: Union["Path", str], replace: bool = True):
        """Copy the current file to destination.

        Args:
            dst (Path | str): Destination path or directory.
            replace (bool, optional): If True, remove any existing destination
                file. Defaults to True.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
        """
        # Validate inputs
        if not self.exists():
            raise FileNotFoundError(f"{self} does not exist.")

        # Normalize inputs
        dst = Path(dst)

        # If dst is an existing directory, copy inside it
        if dst.is_dir():
            final_dst = dst / self.name
        else:
            final_dst = dst

        # Create a parent dir if missing
        final_dst.parent.mkdir(parents=True, exist_ok=True)

        if replace or not final_dst.exists():
            shutil.copy2(str(self), str(final_dst)) # copy2 preserves metadata

        return final_dst

    def rmdir(self, recursive: bool = True):
        """Remove the directory.

        Args:
            recursive (bool, optional): If True, use ``shutil.rmtree`` for
                recursive deletion. Defaults to True.
        """
        if not self.exists():
            return
        if self.is_file():
            self.unlink()
        elif recursive:
            shutil.rmtree(self)
        else:
            super().rmdir()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
