#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Serialize and deserialize data to file (yaml, txt, json, ...).
"""

from __future__ import annotations

import inspect
import json
import pickle
import sys
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import TextIO
from typing import Union

import numpy as np
import xmltodict
import yaml

from one.core.factory import FILE_HANDLERS
from one.core.types import assert_dict
from one.core.types import assert_dict_contain_key
from one.core.types import ScalarOrSequenceT

try:
    from yaml import CLoader as FullLoader, CDumper as Dumper
except ImportError:
    from yaml import FullLoader, Dumper
    

# MARK: - Functional

def dump_file(
    obj        : Any,
    path       : Union[str, Path, TextIO],
    file_format: Union[str, None] = None,
    **kwargs
) -> Union[bool, str]:
    """Dump data to json/yaml/pickle strings or files. This method provides a
    unified api for dumping data as strings or to files, and also supports
    custom arguments for each file format.
    
    Args:
        obj (any):
            Python object to be dumped.
        path (str, Path, TextIO):
            If not specified, then the object is dump to a str, otherwise to a
            file specified by the filename or file-like object.
        file_format (str, None):
            If not specified, the file format will be inferred from the file
            extension, otherwise use the specified one. Currently, supported
            formats include "json", "yaml/yml" and "pickle/pkl". Default: `None`.
    
    Returns:
        (bool, str):
            `True` for success, `False` otherwise.
    """
    if isinstance(path, Path):
        path = str(path)
    if file_format is None:
        if isinstance(path, str):
            file_format = path.split(".")[-1]
        elif path is None:
            raise ValueError("`file_format` must be specified since file is `None`.")

    assert_dict_contain_key(FILE_HANDLERS, file_format)
    
    handler = FILE_HANDLERS.build(name=file_format)
    if path is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(path, str):
        handler.dump_to_file(obj, path, **kwargs)
    elif hasattr(path, "write"):
        handler.dump_to_fileobj(obj, path, **kwargs)
    else:
        raise TypeError("`path` must be a filename str or a file-object.")
    
    
def load_file(
    path       : Union[str, Path, TextIO],
    file_format: Union[str] = None,
    **kwargs
) -> Union[Union[str, dict], None]:
    """Load data from json/yaml/pickle files. This method provides a unified
    api for loading data from serialized files.
   
    Args:
        path (str, Path, TextIO):
            Filename, path, or a file-like object.
        file_format (str, None):
            If not specified, the file format will be inferred from the file
            extension, otherwise use the specified one. Currently, supported
            formats include "json", "yaml/yml" and "pickle/pkl". Default: `None`.
   
    Returns:
        data (str, dict, None):
            Content from the file.
    """
    if isinstance(path, Path):
        path = str(path)
    if file_format is None and isinstance(path, str):
        file_format = path.split(".")[-1]
    
    assert_dict_contain_key(FILE_HANDLERS, file_format)

    handler = FILE_HANDLERS.build(name=file_format)
    if isinstance(path, str):
        data = handler.load_from_file(path, **kwargs)
    elif hasattr(path, "read"):
        data = handler.load_from_fileobj(path, **kwargs)
    else:
        raise TypeError("`file` must be a filepath str or a file-object.")
    return data


def merge_files(
    in_paths   : ScalarOrSequenceT[Union[str, Path, TextIO]],
    out_path   : Union[str, Path, TextIO],
    file_format: Union[str, None] = None,
    **kwargs
) -> Union[bool, str]:
    """Merge data from json/yaml/pickle strings or files.
    
    Args:
        in_paths (ScalarOrSequenceT[str, Path, TextIO]):
            Paths to join.
        out_path (str, Path, TextIO):
            If not specified, then the object is dump to a str, otherwise to a
            file specified by the filename or file-like object.
        file_format (str, None):
            If not specified, the file format will be inferred from the file
            extension, otherwise use the specified one. Currently, supported
            formats include "json", "yaml/yml" and "pickle/pkl". Default: `None`.
    
    Returns:
        (bool, str):
            `True` for success, `False` otherwise.
    """
    if isinstance(in_paths, (str, Path, TextIO)):
        in_paths = [in_paths]
    in_paths = [str(p) for p in in_paths]
    
    # Read data
    data     = None
    for p in in_paths:
        d = load_file(path=p)
        if isinstance(d, list):
            data = [] if data is None else data
            data += d
        elif isinstance(d, dict):
            data = {} if data is None else data
            data |= d
        else:
            raise TypeError(
                f"Input value must be a `list` or `dict`. But got: {type(d)}."
            )
    
    # Dump data
    return dump_file(obj=data, path=out_path, file_format=file_format)
    

# MARK: - Modules

class BaseFileHandler(metaclass=ABCMeta):
    """Base file handler implements the template methods (i.e., skeleton) for
    read and write data from/to different file formats.
    """
    
    @abstractmethod
    def load_from_fileobj(
        self, path: Union[str, TextIO], **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load the content from the given filepath or file-like object (input stream).
        """
        pass
        
    @abstractmethod
    def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
        """Dump data from the given obj to the filepath or file-like object.
        """
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs) -> str:
        """Dump data from the given obj to string."""
        pass

    def load_from_file(
        self, path: str, mode: str = "r", **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load content from the given file."""
        with open(path, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_file(self, obj, path: str, mode: str = "w", **kwargs):
        """Dump data from object to file.
        
        Args:
            obj:
                Object.
            path (str):
                Filepath.
            mode (str):
                File opening mode.
        """
        with open(path, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


@FILE_HANDLERS.register(name="json")
class JsonHandler(BaseFileHandler):
    """JSON file handler."""
    
    @staticmethod
    def set_default(obj):
        """Set default json values for non-serializable values. It helps
        convert `set`, `range` and `np.ndarray` data types to list. It also
        converts `np.generic` (including `np.int32`, `np.float32`, etc.) into
        plain numbers of plain python built-in types.
        """
        if isinstance(obj, (set, range)):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"{type(obj)} is not supported for json dump.")
    
    def load_from_fileobj(
        self, path: Union[str, TextIO], **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load the content from the given filepath or file-like object
        (input stream).
        """
        return json.load(path)

    def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
        """Dump data from the given obj to the filepath or file-like object.
        """
        kwargs.setdefault("default", self.set_default)
        json.dump(obj, path, **kwargs)

    def dump_to_str(self, obj, **kwargs) -> str:
        """Dump data from the given obj to string."""
        kwargs.setdefault("default", self.set_default)
        return json.dumps(obj, **kwargs)


@FILE_HANDLERS.register(name="pickle")
@FILE_HANDLERS.register(name="pkl")
class PickleHandler(BaseFileHandler):
    """Pickle file handler."""
    
    def load_from_fileobj(
        self, path: Union[str, TextIO], **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load the content from the given filepath or file-like object
        (input stream).
        """
        return pickle.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
        """Dump data from the given obj to the filepath or file-like object.
        """
        kwargs.setdefault("protocol", 4)
        pickle.dump(obj, path, **kwargs)
        
    def dump_to_str(self, obj, **kwargs) -> bytes:
        """"Dump data from the given obj to string."""
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)
        
    def load_from_file(
        self, file: Union[str, Path], **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load content from the given file."""
        return super().load_from_file(file, mode="rb", **kwargs)
    
    def dump_to_file(self, obj, path: Union[str, Path], **kwargs):
        """Dump data from object to file.
        
        Args:
            obj:
                Object.
            path (str, Path):
                Filepath.
        """
        super().dump_to_file(obj, path, mode="wb", **kwargs)


@FILE_HANDLERS.register(name="xml")
class XmlHandler(BaseFileHandler):
    """XML file handler."""
    
    def load_from_fileobj(
        self, path: Union[str, TextIO], **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load data from file object (input stream).

        Args:
            path (str, TextIO):
                Filepath or a file-like object.

        Returns:
            (str, dict, None):
                Content from the file.
        """
        doc = xmltodict.parse(path.read())
        return doc

    def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
        """Dump data from obj to file.

        Args:
            obj:
                Object.
            path (str, TextIO):
                Filepath or a file-like object.
        """
        assert_dict(obj)
        with open(path, "w") as path:
            path.write(xmltodict.unparse(obj, pretty=True))
        
    def dump_to_str(self, obj, **kwargs) -> str:
        """Dump data from obj to string.

        Args:
            obj:
                Object.

        Returns:
            (str):
                Content from the file.
        """
        assert_dict(obj)
        return xmltodict.unparse(obj, pretty=True)


@FILE_HANDLERS.register(name="yaml")
@FILE_HANDLERS.register(name="yml")
class YamlHandler(BaseFileHandler):
    """YAML file handler."""
    
    def load_from_fileobj(
        self, path: Union[str, TextIO], **kwargs
    ) -> Union[Union[str, dict], None]:
        """Load the content from the given filepath or file-like object
        (input stream).
        """
        kwargs.setdefault("Loader", FullLoader)
        return yaml.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Union[str, TextIO], **kwargs):
        """Dump data from the given obj to the filepath or file-like object.
        """
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, path, **kwargs)

    def dump_to_str(self, obj, **kwargs) -> str:
        """Dump data from the given obj to string."""
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
