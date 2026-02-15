#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Path.

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
from typing import Iterable, Optional

from mon.core.enum import (
    ConfigExtension,
    ImageExtension,
    VideoExtension,
    WeightExtension,
)


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Path(type(Path_())):  # Dynamic inheritance based on OS
    """Custom Path."""

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
    def ckpt_file(self) -> "Path":
        """Return a matching .ckpt file if present."""
        return self.sibling(".ckpt")

    @property
    def onnx_file(self) -> "Path":
        """Return a matching .onnx file if present."""
        return self.sibling(".onnx")

    # --- Discovery ---
    def files(self, pattern: str = "*", recursive: bool = False) -> list["Path"]:
        """List files in this directory. Safe against non-existent paths.

        Args:
            pattern (str, optional): Glob pattern to filter files. Defaults to "*".
            recursive (bool, optional): If True, include files in subdirectories.
                Defaults to False.
        """
        root = self._get_search_root()
        if not root.exists():
            return []

        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        return [p for p in iterator if p.is_file()]

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
        files = self.files()
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
    def has_ext(self, *extensions: str | Iterable[str], exist: bool = False) -> bool:
        """Robust extension check.

        Args:
            *extensions (str | Iterable[str]): List of extensions to check.
            exist (bool, optional): If True, check if the file exists.
                Defaults to False.
        """
        if exist and not self.is_file():
            return False

        # Flatten inputs
        exts = set()
        for item in extensions:
            if isinstance(item, (set, list, tuple)):
                exts.update(item)
            else:
                exts.add(item)
        # Normalize to lowercase
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

        return self.suffix.lower() in exts

    def is_image_file(self, exist: bool = True) -> bool:
        """Check if the path has a recognized image extension.

        Args:
            exist (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(ImageExtension, exist=exist)

    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Check if the path is a raw image format.

        Args:
            exist (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext({".dng", ".arw"}, exist=exist)

    def is_video_file(self, exist: bool = True) -> bool:
        """Check if the path is a recognized video file.

        Args:
            exist (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(VideoExtension, exist=exist)

    def is_weights_file(self, exist: bool = True) -> bool:
        """Check if the path matches known weight file extensions.

        Args:
            exist (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(WeightExtension, exist=exist)

    def is_config_file(self, exist: bool = True) -> bool:
        """Check if the path matches known config extensions.

        Args:
            exist (bool, optional): If True, also check if the file exists.
                Defaults to True.
        """
        return self.has_ext(ConfigExtension, exist=exist)

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
        if self.name == dirname:
            return self
        return self / dirname

    # --- Computation ---
    def commonpath_to(self, other: "Path") -> "Path":
        """Return the longest common path prefix between two paths.

        Args:
            other (Path): Another path to compare with.
        """
        return Path(os.path.commonpath([str(self), str(other)]))

    def relative_path_to(self, start_part: str) -> "Path":
        """Return a new Path starting from the first occurrence of ``start_part``.

        Example: Path("/a/b/c/d").relative_path_to("b") -> Path("b/c/d")

        Args:
            start_part (str): Substring to start the new relative path from.
        """
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
        new_name = self.name.replace(old, new)
        return self.with_name(new_name)

    # --- Filesystem ---
    def copy_to(self, dst: "Path" | str, replace: bool = True):
        """Copy the current file to destination.

        Args:
            dst (Path | str): Destination path or directory.
            replace (bool, optional): If True, remove any existing destination
                file. Defaults to True.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
        """
        if not self.exists():
            raise FileNotFoundError(f"{self} does not exist.")

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


    # --- Discovery ---
    def data_dir(self) -> Optional["Path"]:
        """Go up the directory tree to find a ``data`` directory."""
        current_dir = self.parent if self.is_file() else self
        for parent in current_dir.parents:
            if (parent / "data").exists():
                return parent / "data"
        return None

    def image_dirs(self, recursive: bool = False) -> list["Path"]:
        """Return an iterator over image directories under this path.

        Args:
            recursive (bool): If True, include nested subdirectories.
                Defaults to False.
        """
        root = self.parent if self.is_file() else self
        search_paths = root.rglob("image") if recursive else root.iterdir()
        return [p for p in search_paths if p.is_dir() and p.stem == "image"]

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
