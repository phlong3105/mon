#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements an XML file handler by extending the :mod:`xmltodict`
module.
"""

from __future__ import annotations

from typing import Any, TextIO

import munch
from xmltodict import *

from mon.core import constant, pathlib
from mon.core.file_handler import base
from mon.core.typing import PathType


# region XML File Handler

@constant.FILE_HANDLER.register(name=".xml")
class XMLHandler(base.FileHandler):
    """XML file handler."""

    def read_from_fileobj(self, path: PathType | TextIO, **kwargs) -> Any:
        """Load content from a file.

        Args:
            path: A filepath or an input stream object.

        Returns:
            File's content.
        """
        doc = parse(path.read())
        return doc

    def write_to_fileobj(self, obj: Any, path: PathType | TextIO, **kwargs):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: The filepath to dump :param:`obj` content.
        """
        path = pathlib.Path(path)
        assert isinstance(obj, dict | munch.Munch)
        with open(path, "w") as path:
            path.write(unparse(obj, pretty=True))

    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a string.

        Args:
            obj: A serializable object.
        """
        assert isinstance(obj, dict | munch.Munch)
        return unparse(obj, pretty=True)

# endregion
