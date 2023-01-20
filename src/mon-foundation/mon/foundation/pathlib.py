#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python :mod:`pathlib` module."""

from __future__ import annotations

__all__ = [
    "Path", "PosixPath", "PurePath", "PurePosixPath", "PureWindowsPath",
    "WindowsPath", "is_basename", "is_bmp_file", "is_ckpt_file",
    "is_image_file", "is_json_file", "is_name", "is_stem", "is_torch_file",
    "is_txt_file", "is_url", "is_url_or_file", "is_video_file",
    "is_video_stream", "is_weights_file", "is_xml_file", "is_yaml_file",
]

import pathlib
from pathlib import *
from typing import TYPE_CHECKING

import validators

from mon.foundation import constant

if TYPE_CHECKING:
    from mon.foundation.typing import PathType


# region Path

class Path(type(pathlib.Path())):
    """An extension of :class:`pathlib.Path` with more functionalities."""
    
    def is_basename(self) -> bool:
        """Return True if the current path in :attr::`self` is a basename of a
        file. Otherwise, return False.
        """
        return str(self) == self.name
    
    def is_bmp_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a bitmap file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() == constant.ImageFormat.BMP.value
    
    def is_ckpt_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a checkpoint
        file. Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in [".ckpt"]
    
    def is_image_file(self) -> bool:
        """Return True if the current path in :attr::`self` is an image file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in constant.ImageFormat.values()
    
    def is_json_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a JSON file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in [".json"]
    
    def is_name(self) -> bool:
        """Return True if the current path in :attr::`self` is the same as the
        stem. Otherwise, return False.
        """
        return self == self.stem
    
    def is_stem(self) -> bool:
        """Return True if the current path in :attr::`self` isn't None, and the
        parent of the path is the current directory, and the path has no
        extension. Otherwise, return False.
        """
        return str(self) == self.stem
    
    def is_torch_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a file, and the
        file extension is one of the following:
            - pt
            - pt.tar
            - pth
            - pth.tar
            - weights
            - ckpt
            Otherwise, return False.
        """
        return self.is_file() \
            and self.suffix.lower() in [
                ".pt", ".pth", ".weights", ".ckpt", ".tar"
            ]
    
    def is_txt_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a text file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in [".txt"]
    
    def is_url(self) -> bool:
        """Return True if the current path in :attr::`self` is a valid URL.
        Otherwise, return False.
        """
        return not isinstance(validators.url(str(self)), validators.ValidationFailure)
    
    def is_url_or_file(self) -> bool:
        """Return True if the path is a file or a valid URL. Otherwise,
        return False.
        """
        return self.is_file() or \
            not isinstance(validators.url(self), validators.ValidationFailure)
    
    def is_video_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a video file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in constant.VideoFormat.values()
    
    def is_video_stream(self) -> bool:
        """Return True if the current path in :attr::`self` is a video stream.
        Otherwise, return False.
        """
        return "rtsp" in str(self).lower()
    
    def is_weights_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a PyTorch's
        weight file. Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in [".pt", ".pth"]
    
    def is_xml_file(self) -> bool:
        """Return True if the current path in :attr::`self` is an XML file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in [".xml"]
    
    def is_yaml_file(self) -> bool:
        """Return True if the current path in :attr::`self` is a YAML file.
        Otherwise, return False.
        """
        return self.is_file() and self.suffix.lower() in [".yaml", ".yml"]


def is_basename(path: PathType) -> bool:
    """Return True if a path is a basename of a file. Otherwise, return False.
    """
    path = Path(path)
    return str(path) == path.name


def is_bmp_file(path: PathType) -> bool:
    """Return True if a path is a bitmap file. Otherwise, return False."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() == constant.ImageFormat.BMP.value


def is_ckpt_file(path: PathType) -> bool:
    """Return True if a path is a .ckpt file. Otherwise, return False."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".ckpt"]


def is_image_file(path: PathType) -> bool:
    """Return True if the path is an image file. Otherwise, return False.
    """
    path = Path(path)
    return path.is_file() and path.suffix.lower() in constant.ImageFormat.values()


def is_json_file(path: PathType) -> bool:
    """Return True if a path is a JSON file. Otherwise, return False."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".json"]


def is_name(path: PathType) -> bool:
    """Return True if a path is the same as the stem. Otherwise, return False.
    """
    path = Path(path)
    return path == path.stem


def is_stem(path: PathType) -> bool:
    """Return True if a path isn't None, and the parent of the path is the
    current directory, and the path has no extension. Otherwise, return False.
    """
    path = Path(path)
    return str(path) == path.stem


def is_torch_file(path: PathType) -> bool:
    """Return True If a path is a file, and the file extension is one of:
        - pt
        - pt.tar
        - pth
        - pth.tar
        - weights
        - ckpt
        Otherwise, return False.
    """
    path = Path(path)
    return path.is_file() \
        and path.suffix.lower() in [".pt", ".pth", ".weights", ".ckpt", ".tar"]


def is_txt_file(path: PathType) -> bool:
    """Return True if a path is a text file. Otherwise, return False."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".txt"]


def is_url(path: PathType) -> bool:
    """Return True if a path is a valid URL. Otherwise, return False."""
    path = Path(path)
    return not isinstance(validators.url(str(path)), validators.ValidationFailure)


def is_url_or_file(path: PathType) -> bool:
    """Return True if a path is a file or a valid URL. Otherwise, return False.
    """
    path = Path(path)
    return path.is_file() or \
        not isinstance(validators.url(path), validators.ValidationFailure)


def is_video_file(path: PathType) -> bool:
    """Return True if a path is a video file. Otherwise, return False."""
    path = Path(path)
    return path.is_file() and path.suffix.lower() in constant.VideoFormat.values()


def is_video_stream(path: PathType) -> bool:
    """Return True if a path is a video stream. Otherwise, return False."""
    path = Path(path)
    return "rtsp" in str(path).lower()


def is_weights_file(path: PathType) -> bool:
    """Return True if a path is a PyTorch's weight file. Otherwise, return
    False.
    """
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".pt", ".pth"]


def is_xml_file(path: PathType) -> bool:
    """Return True if a path is an XML file. Otherwise, return False. """
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".xml"]


def is_yaml_file(path: PathType) -> bool:
    """Return True if a path is a YAML file. Otherwise, return False. """
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".yaml", ".yml"]

# endregion
