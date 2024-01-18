#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a Pickle file handler by extending the :mod:`pickle`
module.
"""

from __future__ import annotations

from pickle import *
from typing import Any, TextIO

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region Pickle File Handler

@FILE_HANDLERS.register(name=".pickle")
@FILE_HANDLERS.register(name=".pkl")
class PickleHandler(base.FileHandler):
    """Pickle file handler."""
    
    def read_from_fileobj(
        self,
        path: pathlib.Path | str | TextIO,
        **kwargs
    ) -> Any:
        """Load content from a file.

        Args:
            path: A file path or an input stream object.

        Returns:
            File's content.
        """
        path = pathlib.Path(path)
        return load(path, **kwargs)
    
    def write_to_fileobj(
        self,
        obj : Any,
        path: pathlib.Path | str | TextIO,
        **kwargs
    ):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: The file path to dump :param:`obj` content.
        """
        path = pathlib.Path(path)
        kwargs.setdefault("protocol", 4)
        dump(obj, path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a :class:`str`.

        Args:
            obj: A serializable object.
        """
        kwargs.setdefault("protocol", 2)
        return dumps(obj, **kwargs)
    
    def read_from_file(
        self,
        path: pathlib.Path | str,
        mode: str = "r",
        **kwargs
    ) -> Any:
        """Load content from a file.

        Args:
            path: A file path.
            mode: The mode to open the file. Default: ``'r'`` means read.

        Returns:
            File's content.
        """
        path = pathlib.Path(path)
        return super().read_from_file(path=path, mode="rb", **kwargs)
    
    def write_to_file(
        self,
        obj : Any,
        path: pathlib.Path | str,
        mode: str = "w",
        **kwargs
    ):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: A file path.
            mode: The mode to open the file. Default: ``'w'`` means write.
        """
        path = pathlib.Path(path)
        super().write_to_file(obj=obj, path=path, mode="wb", **kwargs)

# endregion
