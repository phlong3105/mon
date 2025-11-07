#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the standard ``pathlib.Path`` class with additional
functionalities.
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


# ----- Path Class -----
class Path(type(Path_())):
    """An extension of ``pathlib.Path`` with additional functionalities.
    
    Notes:
        Methods are kept as methods (not properties) for consistency with ``pathlib.Path``.
    """
    
    # ----- Properties -----
    def hash(self) -> int:
        """Calculates the hash value of the file based on its size.

        Returns:
            Integer hash value of the file size.
        """
        return self.stat().st_size if self.is_file() else 0
    
    # ----- Check Internal Parts -----
    def is_basename(self) -> bool:
        """Checks if the path is a file basename.

        Returns:
            ``True`` if path matches its basename, ``False`` otherwise.
        """
        return str(self) == self.name
     
    def is_name(self) -> bool:
        """Checks if the path matches its stem.

        Returns:
            ``True`` if path equals its stem, ``False`` otherwise.
        """
        return str(self) == self.stem
    
    def is_stem(self) -> bool:
        """Checks if the path matches its stem.

        Returns:
            ``True`` if path equals its stem, ``False`` otherwise.
        """
        return str(self) == self.stem
    
    def is_url(self) -> bool:
        """Checks if the path is a valid URL.

        Returns:
            ``True`` if path is a valid URL, ``False`` otherwise.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationError)
    
    def is_url_or_file(self, exist: bool = True) -> bool:
        """Checks if the path is a file or valid URL.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a file or valid URL, ``False`` otherwise.
        """
        return (
            (not exist or self.is_file())
            or not isinstance(validators.url(str(self)), validators.ValidationError)
        )
    
    def is_file_like(self) -> bool:
        """"Checks if the path resembles a file format.

        Returns:
            ``True`` if path has a suffix, ``False`` otherwise.
        """
        return "." in self.suffix
    
    def is_dir_like(self) -> bool:
        """Checks if the path resembles a directory format.

        Returns:
            ``True`` if path has no suffix, ``False`` otherwise.
        """
        return self.suffix == ""
    
    def has_subdir(self, name: str) -> bool:
        """Checks if the directory has a subdirectory with the given name.

        Args:
            name: Subdirectory name to check.

        Returns:
            ``True`` if subdirectory exists, ``False`` otherwise.
        """
        return name in [d.name for d in self.subdirs()]
    
    # ----- Check Text File -----
    def is_json_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.json`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.json`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".json"
    
    def is_txt_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.txt`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.txt`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".txt"
    
    def is_xml_file(self, exist: bool = True) -> bool:
        """Checks if the path is an ``.xml`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is an ``.xml`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".xml"
    
    def is_yaml_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.yaml`` or ``.yml`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.yaml`` or ``.yml`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".yaml", ".yml"]
   
    # ----- Check Image File -----
    def is_image_file(self, exist: bool = True) -> bool:
        """Checks if the path is an image file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is an image file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in ImageExtension
        
    def is_raw_image_file(self, exist: bool = True) -> bool:
        """Checks if the path is a raw image file (``.dng`` or ``.arw``).

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a raw image file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in [".dng", ".arw"]
    
    # ----- Check Video File -----
    def is_video_file(self, exist: bool = True) -> bool:
        """Checks if the path is a video file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a video file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in VideoExtension
    
    def is_video_stream(self) -> bool:
        """Checks if the path is a video stream.

        Returns:
            ``True`` if path contains ``rtsp``, ``False`` otherwise.
        """
        return "rtsp" in str(self).lower()
    
    # ----- Check ML File -----
    def is_cache_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.cache`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.cache`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".cache"
    
    def is_ckpt_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.ckpt`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.ckpt`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".ckpt"
    
    def is_config_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.config`` or ``.cfg`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a config file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in ConfigExtension

    def is_onnx_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.onnx`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.onnx`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".onnx"

    def is_py_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.py`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a ``.py`` file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() == ".py"

    def is_weights_file(self, exist: bool = True) -> bool:
        """Checks if the path is a ``.pt`` or ``.pth`` file.

        Args:
            exist: If ``True``, verifies the file exists. Default: ``True``.

        Returns:
            ``True`` if path is a weights file, ``False`` otherwise.
        """
        return (not exist or self.is_file()) and self.suffix.lower() in WeightExtension
    
    # ----- List -----
    def subdirs(self, recursive: bool = False) -> list["Path"]:
        """Returns a list of subdirectory paths.

        Args:
            recursive: If ``True``, includes subdirs recursively. Default: ``False``.

        Returns:
            List of subdirectory paths.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_dir()]
    
    def files(self, recursive: bool = False) -> list["Path"]:
        """Returns a list of file paths in the directory.

        Args:
            recursive: If ``True``, includes files in subdirs. Default: ``False``.

        Returns:
            List of file paths.
        """
        path = self.parent if self.is_file_like() else self
        paths = list(path.rglob("*")) if recursive else list(path.iterdir())
        return [p for p in paths if p.is_file()]
    
    def ckpt_file(self) -> "Path":
        """Returns the checkpoint file path if found.

        Returns:
            Checkpoint file path or ``None`` if not found.
        """
        ckpt_path = self.with_suffix(".ckpt")
        return ckpt_path if ckpt_path.is_file() else self
    
    def config_file(self) -> "Path":
        """Returns the configuration file path.

        Returns:
            Configuration file path.
        """
        for ext in ConfigExtension.values():
            for stem in [self.stem, snakecase(self.stem)]:
                config_path = self.with_name(f"{stem}{ext}")
                if config_path.is_file():
                    return config_path
        return self

    def label_file(self) -> "Path":
        """Returns the label file path."""
        for ext in [".txt", ".xml", ".json"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def latest_file(self) -> "Path":
        """Returns the latest file based on creation time.

        Returns:
            Latest file path or ``None`` if no files exist.
        """
        files = self.files()
        return max(files, key=os.path.getctime) if files else None
    
    def image_file(self) -> "Path":
        """Returns the image file path."""
        for ext in ImageExtension.values():
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def txt_file(self) -> "Path":
        """Returns the .txt file path."""
        for ext in [".txt"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self

    def yaml_file(self) -> "Path":
        """Returns the YAML file path."""
        for ext in [".yaml", ".yml"]:
            temp = self.with_suffix(ext)
            if temp.is_file():
                return temp
        return self
    
    def relative_path(self, start_part: str) -> "Path":
        """Returns the relative path from a given start part.

        Args:
            start_part: Starting path or string for relativity.

        Returns:
            Relative path from ``start_part``.
        """
        path       = Path(self)
        start_part = str(start_part)
        path_str   = str(path)
        if start_part not in path_str:
            return path
        start_idx = path_str.index(start_part)
        return Path(path_str[start_idx:])
    
    # ----- Creation -----
    def copy_to(self, dst: str, replace: bool = True):
        """Copies the file to a new location.

        Args:
            dst: Destination path or string.
            replace: If ``True``, replaces the existing file. Default: ``True``.

        Raises:
            NotImplementedError: If ``dst`` is a URL.
        """
        dst = Path(dst)
        if dst.is_url():
            raise NotImplementedError("[dst] as a URL is not supported.")
        dst = dst / self.name if dst.is_dir_like() else dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if replace:
            dst.unlink(missing_ok=True)
        shutil.copyfile(src=str(self), dst=str(dst))
    
    def replace_part(self, old: str, new: str, count: int = 1) -> "Path":
        """Replaces part of the Path.

        Args:
            old: String to replace.
            new: Replacement string.
            count: Max number of replacements. Default: ``1``.

        Returns:
            New path with replaced string.
        """
        return Path(str(self).replace(old, new, count))
    
    # ----- Deletion -----
    def rmdir(self):
        """Removes the directory and its contents."""
        delete_files(path=self, regex="*", recursive=True)
        super().rmdir()
        

# ----- Download -----
def download_url_to_file(url: str, path: Path, overwrite: bool = False) -> Path:
    """Downloads weights from a ``url`` to a local ``path``.

    Args:
        url: The URL to download weights from.
        path: The local file path to save weights.
        overwrite: If ``True``, overwrites the existing file. Default: ``False``.

    Returns:
        The ``Path`` to downloaded weights file.

    Raises:
        ValueError: If ``url`` is not a valid URL.
    """
    if not Path(url).is_url():
        raise ValueError(f"``url`` must be a valid URL, got {url}.")
    
    path = Path(path)
    if not path.exists() or overwrite:
        path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.hub.download_url_to_file(url, path, None, True)
    return path


# ----- Delete -----
def delete_files(path: Path, regex: str = None, recursive: bool = False):
    """Deletes files matching a pattern in a directory.

    Args:
        path: Directory path to search for files.
        regex: File path pattern. Default: ``None`` (deletes ``path`` if file).
        recursive: If ``True``, searches subdirs. Default: ``False``.
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
            print(f"Cannot delete file: [{err}].")
