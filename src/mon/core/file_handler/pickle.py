#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a Pickle file handler by extending the :mod:`pickle`
module.
"""

from __future__ import annotations

from pickle import *
from typing import Any, TextIO

from mon.core import constant, pathlib
from mon.core.file_handler import base
from mon.core.typing import PathType


# region Pickle File Handler

@constant.FILE_HANDLER.register(name=".pickle")
@constant.FILE_HANDLER.register(name=".pkl")
class PickleHandler(base.FileHandler):
    """Pickle file handler."""

    def read_from_fileobj(self, path: PathType | TextIO, **kwargs) -> Any:
        """Load content from a file.

        Args:
            path: A filepath or an input stream object.

        Returns:
            File's content.
        """
        path = pathlib.Path(path)
        return load(path, **kwargs)

    def write_to_fileobj(self, obj: Any, path: PathType | TextIO, **kwargs):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: The filepath to dump :param:`obj` content.
        """
        path = pathlib.Path(path)
        kwargs.setdefault("protocol", 4)
        dump(obj, path, **kwargs)

    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a string.

        Args:
            obj: A serializable object.
        """
        kwargs.setdefault("protocol", 2)
        return dumps(obj, **kwargs)

    def read_from_file(self, path: PathType, mode: str = "r", **kwargs) -> Any:
        """Load content from a file.

        Args:
            path: A filepath.
            mode: The mode to open the file. Defaults to “r” means read.

        Returns:
            File's content.
        """
        path = pathlib.Path(path)
        return super().read_from_file(path, mode="rb", **kwargs)

    def write_to_file(self, obj: Any, path: PathType, mode: str = "w", **kwargs):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: A filepath.
            mode: The mode to open the file. Defaults to “w” means write.
        """
        path = pathlib.Path(path)
        super().write_to_file(obj, path, mode="wb", **kwargs)

# endregion
