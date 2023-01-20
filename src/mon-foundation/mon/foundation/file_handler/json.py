#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a JSON file handler by extending the :mod:`json`
package.
"""

from __future__ import annotations

from json import *
from typing import Any, TextIO

import numpy as np

from mon.foundation import constant, pathlib
from mon.foundation.file_handler import base
from mon.foundation.typing import PathType


# region JSON File Handler

@constant.FILE_HANDLER.register(name=".json")
class JSONHandler(base.FileHandler):
    """JSON file handler."""
    
    @staticmethod
    def set_default(obj: Any):
        """If an object is a set, range, numpy array, or numpy generic, convert
        it to a list.
        
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
    def load_from_fileobj(self, path: PathType | TextIO, **kwargs) -> Any:
        """Load content from a file.

        Args:
            path: A filepath or an input stream object.

        Returns:
            File's content.
        """
        return load(path)

    # noinspection PyTypeChecker
    def dump_to_fileobj(self, obj: Any, path: PathType | TextIO, **kwargs):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: The filepath to dump :param:`obj` content.
        """
        path = pathlib.Path(path)
        kwargs.setdefault("default", self.set_default)
        dump(obj, path, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a string.

        Args:
            obj: A serializable object.
        """
        kwargs.setdefault("default", self.set_default)
        return dumps(obj, **kwargs)

# endregion
