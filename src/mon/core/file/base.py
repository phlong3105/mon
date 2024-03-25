#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all file handlers, and related
helper functions.
"""

from __future__ import annotations

__all__ = [
    "FileHandler",
    "write_to_file",
    "read_from_file",
    "merge_files",
]

from abc import ABC, abstractmethod
from typing import Any, TextIO

from mon.core import dtype, pathlib
from mon.globals import FILE_HANDLERS


# region File Handler

class FileHandler(ABC):
    """The base class for reading and writing data from/to different file
    formats.
    """
    
    @abstractmethod
    def read_from_fileobj(self, path: pathlib.Path | str | TextIO, **kwargs) -> Any:
        """Load content from a file.
        
        Args:
            path: A file path or an input stream object.
        
        Returns:
            File's content.
        """
        pass
    
    @abstractmethod
    def write_to_fileobj(self, obj: Any, path: pathlib.Path | str | TextIO, **kwargs):
        """Write content from a serializable object to a file.
        
        Args:
            obj: A serializable object.
            path: The file path to Write :param:`obj` content.
        """
        pass
    
    @abstractmethod
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Write content from a serializable object to a string.
        
        Args:
            obj: A serializable object.
        """
        pass
    
    def read_from_file(self, path: pathlib.Path | str, mode: str = "r", **kwargs) -> Any:
        """Load content from a file.
        
        Args:
            path: A file path.
            mode: The mode to open the file. Default: ``'r'`` means read.
        
        Returns:
            File's content.
        """
        with open(path, mode) as f:
            return self.read_from_fileobj(path=f, **kwargs)
    
    def write_to_file(self, obj: Any, path: pathlib.Path | str, mode: str = "w", **kwargs):
        """Write content from a serializable object to a file.
        
        Args:
            obj: A serializable object.
            path: A file path.
            mode: The mode to open the file. Default: ``'w'`` means write.
        """
        with open(path, mode) as f:
            self.write_to_fileobj(obj=obj, path=f, **kwargs)


def write_to_file(
    obj        : Any,
    path       : pathlib.Path | str | TextIO,
    file_format: str | None = None,
    **kwargs
):
    """Write content from a serializable object to a file.
    
    Args:
        obj: A serializable object.
        path: A file path.
        file_format: The file format. If not specified, it is inferred from the
            :param:`path`'s extension. Default: ``None``.
    """
    path = pathlib.Path(path)
    if file_format is None:
        file_format = path.suffix
    if file_format not in FILE_HANDLERS:
        raise ValueError(
            f"file_format must be a valid key in {FILE_HANDLERS.keys}, but got "
            f"{file_format}."
        )
    
    handler: FileHandler = FILE_HANDLERS.build(name=file_format)
    if path is None:
        handler.write_to_string(obj=obj, **kwargs)
    elif isinstance(path, str):
        handler.write_to_file(obj=obj, path=path, **kwargs)
    elif hasattr(path, "write"):
        handler.write_to_fileobj(obj=obj, path=path, **kwargs)
    else:
        raise TypeError(
            f"path must be a filename or a file-object, but got {type(path)}."
        )


def read_from_file(
    path       : pathlib.Path | str | TextIO,
    file_format: str | None = None,
    **kwargs
) -> Any:
    """Load content from a file.
    
    Args:
        path: A file path.
        file_format: The file format. If not specified, it is inferred from the
            :param:`path`'s extension. Default: ``None``.
    
    Returns:
        File's content.
    """
    path = pathlib.Path(path)
    if file_format is None:
        file_format = path.suffix
    
    handler: FileHandler = FILE_HANDLERS.build(name=file_format)
    if isinstance(path, pathlib.Path | str):
        data = handler.read_from_file(path=path, **kwargs)
    elif hasattr(path, "read"):
        data = handler.read_from_fileobj(path=path, **kwargs)
    else:
        raise TypeError(
            f"'path' must be a 'pathlib.Path', a 'str' or a file-object. "
            f"But got: {type(path)}."
        )
    return data


def merge_files(
    in_paths   : list[pathlib.Path | str | TextIO],
    out_path   : pathlib.Path | str | TextIO,
    file_format: str | None = None,
):
    """Merge content from multiple files to a single file.
    
    Args:
        in_paths: Merging file paths.
        out_path: The output file.
        file_format: The file format. If not specified, it is inferred from the
            :param:`path`'s extension. Default: ``None``.
    """
    in_paths = dtype.to_list(x=in_paths)
    in_paths = [pathlib.Path(p) for p in in_paths]
    
    # Read data
    data = None
    for p in in_paths:
        d = read_from_file(path=p)
        if isinstance(d, list):
            data = [] if data is None else data
            data += d
        elif isinstance(d, dict):
            data = {} if data is None else data
            data |= d
        else:
            raise TypeError(f"Input value must be a :class:`list` or :class:`dict`, but got {type(d)}.")
    
    # Write data
    write_to_file(obj=data, path=out_path, file_format=file_format)

# endregion
