#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements an XML file handler by extending the :mod:`xmltodict`
module.
"""

from __future__ import annotations

from typing import Any, TextIO

from xmltodict import *

from mon.core import pathlib
from mon.core.file import base
from mon.globals import FILE_HANDLERS


# region XML File Handler

@FILE_HANDLERS.register(name=".xml")
class XMLHandler(base.FileHandler):
    """XML file handler."""
    
    def read_from_fileobj(
        self,
        path: pathlib.Path | str | TextIO,
        **kwargs
    ) -> Any:
        """Load content from a file.

        Args:
            path: A filepath or an input stream object.

        Returns:
            File's content.
        """
        doc = parse(path.read())
        return doc
    
    def write_to_fileobj(
        self,
        obj : Any,
        path: pathlib.Path | str | TextIO,
        **kwargs
    ):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: The filepath to dump :param:`obj` content.
        """
        assert isinstance(obj, dict)
        path = pathlib.Path(path)
        with open(path, "w") as f:
            f.write(unparse(input_dict=obj, pretty=True))
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a string.

        Args:
            obj: A serializable object.
        """
        assert isinstance(obj, dict)
        return unparse(input_dict=obj, pretty=True)

# endregion
