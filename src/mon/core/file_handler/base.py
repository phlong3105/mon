#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all file handlers, and related
helper functions.
"""

from __future__ import annotations

__all__ = [
    "FileHandler", "write_to_file", "read_from_file", "merge_files",
]

from abc import ABC, abstractmethod
from typing import Any, Sequence, TextIO

from mon.core import builtins, constant, pathlib
from mon.core.typing import PathType


# region File Handler

class FileHandler(ABC):
    """The base class for reading and writing data from/to different file
    formats.
    """
    
    @abstractmethod
    def read_from_fileobj(self, path: PathType | TextIO, **kwargs) -> Any:
        """Load content from a file.
        
        Args:
            path: A filepath or an input stream object.
        
        Returns:
            File's content.
        """
        pass
        
    @abstractmethod
    def write_to_fileobj(self, obj: Any, path: PathType | TextIO, **kwargs):
        """Dump content from a serializable object to a file.
        
        Args:
            obj: A serializable object.
            path: The filepath to dump :param:`obj` content.
        """
        pass
    
    @abstractmethod
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a string.
        
        Args:
            obj: A serializable object.
        """
        pass

    def read_from_file(self, path: PathType, mode: str = "r", **kwargs) -> Any:
        """Load content from a file.
        
        Args:
            path: A filepath.
            mode: The mode to open the file. Defaults to “r” means read.
        
        Returns:
            File's content.
        """
        with open(path, mode) as f:
            return self.read_from_fileobj(f, **kwargs)

    def write_to_file(self, obj: Any, path: PathType, mode: str = "w", **kwargs):
        """Dump content from a serializable object to a file.
        
        Args:
            obj: A serializable object.
            path: A filepath.
            mode: The mode to open the file. Defaults to “w” means write.
        """
        with open(path, mode) as f:
            self.write_to_fileobj(obj, f, **kwargs)


def write_to_file(
    obj        : Any,
    path       : PathType | TextIO,
    file_format: str      | None = None,
    **kwargs
):
    """Dump content from a serializable object to a file.
    
    Args:
        obj: A serializable object.
        path: A filepath.
        file_format: The file format. If not specified, it is inferred from the
            :param:`path`'s extension. Defaults to None.
    """
    path = pathlib.Path(path)
    if file_format is None:
        file_format = path.suffix
    assert file_format in constant.FILE_HANDLER
    
    handler: FileHandler = constant.FILE_HANDLER.build(name=file_format)
    if path is None:
        handler.write_to_string(obj, **kwargs)
    elif isinstance(path, str):
        handler.write_to_file(obj, path, **kwargs)
    elif hasattr(path, "write"):
        handler.write_to_fileobj(obj, path, **kwargs)
    else:
        raise TypeError(
            f":param:`path` must be a filename or a file-object. "
            f"But got: {type(path)}."
        )


def read_from_file(
    path       : PathType | TextIO,
    file_format: str | None = None,
    **kwargs
) -> Any:
    """Load content from a file.
    
    Args:
        path: A filepath.
        file_format: The file format. If not specified, it is inferred from the
            :param:`path`'s extension. Defaults to None.
    
    Returns:
        File's content.
    """
    path = pathlib.Path(path)
    if file_format is None:
        file_format = path.suffix

    handler: FileHandler = constant.FILE_HANDLER.build(name=file_format)
    if isinstance(path, pathlib.Path | str):
        data = handler.read_from_file(path, **kwargs)
    elif hasattr(path, "read"):
        data = handler.read_from_fileobj(path, **kwargs)
    else:
        raise TypeError(
            f":param:`path` must be a :class:`pathlib.Path`, a :class:`str` or "
            f"a file-object. But got: {type(path)}."
        )
    return data


def merge_files(
    in_paths   : Sequence[PathType | TextIO],
    out_path   : PathType | TextIO,
    file_format: str | None = None,
):
    """Merge content from multiple files to a single file.
    
    Args:
        in_paths: Merging filepaths.
        out_path: The output file.
        file_format: The file format. If not specified, it is inferred from the
            :param:`path`'s extension. Defaults to None.
    """
    in_paths = builtins.to_list(in_paths)
    in_paths = [pathlib.Path(p) for p in in_paths]
    
    # Read data
    data = None
    for p in in_paths:
        d = read_from_file(p)
        if isinstance(d, list):
            data = [] if data is None else data
            data += d
        elif isinstance(d, dict):
            data = {} if data is None else data
            data |= d
        else:
            raise TypeError(
                f"Input value must be a :class:`list` or :class:`dict`. "
                f"But got: {type(d)}."
            )
    
    # Dump data
    write_to_file(
        obj         = data,
        path        = out_path,
        file_format = file_format,
    )

# endregion
