#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements an YAML file handler by extending the :mod:`yaml`
module.
"""

from __future__ import annotations

from typing import Any, TextIO

from yaml import *

from mon.foundation import constant, pathlib
from mon.foundation.file_handler import base
from mon.foundation.typing import PathType


# region YAML File Handler

@constant.FILE_HANDLER.register(name=".yaml")
@constant.FILE_HANDLER.register(name=".yml")
class YAMLHandler(base.FileHandler):
    """YAML file handler."""
    
    def load_from_fileobj(self, path: PathType | TextIO, **kwargs) -> Any:
        """Load content from a file.
        
        Args:
            path: A filepath or an input stream object.
        
        Returns:
            File's content.
        """
        kwargs.setdefault("Loader", FullLoader)
        return load(path, **kwargs)

    def dump_to_fileobj(self, obj: Any, path: PathType | TextIO, **kwargs):
        """Dump content from a serializable object to a file.

        Args:
            obj: A serializable object.
            path: The filepath to dump :param:`obj` content.
        """
        path = pathlib.Path(path)
        kwargs.setdefault("Dumper", Dumper)
        dump(obj, path, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs) -> str:
        """Dump content from a serializable object to a string.

        Args:
            obj: A serializable object.
        """
        kwargs.setdefault("Dumper", Dumper)
        return dump(obj, **kwargs)

# endregion
