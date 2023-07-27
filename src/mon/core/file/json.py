#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a JSON file handler by extending the :mod:`json`
package.
"""

from __future__ import annotations

from json import *
from typing import Any, TextIO

import numpy as np

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region JSON File Handler

@FILE_HANDLERS.register(name=".json")
class JSONHandler(base.FileHandler):
    """JSON file handler."""
    
    @staticmethod
    def set_default(obj: Any):
        """If an object is a :class:`set`, :class:`range`, numpy array, or numpy
        generic, convert it to a :class:`list`.
        
        Args:
            obj: A serializable object.
        """
        if isinstance(obj, set | range):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"{type(obj)} is not supported for json dump.")
    
    # noinspection PyTypeChecker
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
        return load(path)
    
    # noinspection PyTypeChecker
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
        kwargs.setdefault("default", self.set_default)
        dump(obj=obj, fp=path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a :class:`str`.

        Args:
            obj: A serializable object.
        """
        kwargs.setdefault("default", self.set_default)
        return dumps(obj=obj, **kwargs)

# endregion
