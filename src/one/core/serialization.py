#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Serialize and deserialize data to file (yaml, txt, json, ...).
"""

from __future__ import annotations

import inspect
import json
import pathlib
import pickle
import sys
from abc import ABCMeta
from abc import abstractmethod
from typing import Any

import numpy as np
import xmltodict
import yaml

from one.core.factory import FILE_HANDLERS
from one.core.types import assert_dict
from one.core.types import assert_dict_contain_key
from one.core.types import Path
from one.core.types import Paths
from one.core.types import to_list

try:
    from yaml import CLoader as FullLoader, CDumper as Dumper
except ImportError:
    from yaml import FullLoader, Dumper
    

# MARK: - Functional

def dump_file(
    obj        : Any,
    path       : Path,
    file_format: str | None = None,
    **kwargs
) -> bool | str:
    """
    It dumps an object to a file or a file-like object.
    
    Args:
        obj (Any): The object to be dumped.
        path (Path): The path to the file to be written.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        A boolean or a string.
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    if file_format is None:
        if isinstance(path, str):
            file_format = path.split(".")[-1]
        elif path is None:
            raise ValueError(
                "`file_format` must be specified since file is `None`."
            )

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
    path       : Path,
    file_format: str | None = None,
    **kwargs
) -> str | dict | None:
    """
    Load a file from a filepath or file-object, and return the data in the file.
    
    Args:
        path (Path): The path to the file to load.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        The data from the file.
    """
    if isinstance(path, pathlib.Path):
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
    in_paths   : Paths,
    out_path   : Path,
    file_format: str | None = None,
    **kwargs
) -> bool | str:
    """
    Reads data from multiple files and writes it to a single file.
    
    Args:
        in_paths (Paths): The input paths to the files you want to merge.
        out_path (Path): The path to the output file.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        A boolean or a string.
    """
    in_paths = to_list(in_paths)
    in_paths = [str(p) for p in in_paths]
    
    # Read data
    data = None
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
    """
    Base file handler implements the template methods (i.e., skeleton) for
    read and write data from/to different file formats.
    """
    
    @abstractmethod
    def load_from_fileobj(self, path: Path, **kwargs) -> str | dict | None:
        """
        It loads a file from a file object.
        
        Args:
            path (Path): The path to the file to load.
        """
        pass
        
    @abstractmethod
    def dump_to_fileobj(self, obj, path: Path, **kwargs):
        """
        It takes a `self` object, an `obj` object, a `path` object, and a
        `**kwargs` object, and returns nothing.
        
        Args:
            obj: The object to be dumped.
             path (Path): The path to the file to be read.
        """
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It takes an object and returns a string.
        
        Args:
            obj: The object to be serialized.
        """
        pass

    def load_from_file(
        self,
        path: str,
        mode: str = "r",
        **kwargs
    ) -> str | dict | None:
        """
        It loads a file from the given path and returns the contents.
        
        Args:
            path (str): The path to the file to load from.
            mode (str): The mode to open the file in. Defaults to "r".
        
        Returns:
            The return type is a string, dictionary, or None.
        """
        with open(path, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_file(self, obj, path: str, mode: str = "w", **kwargs):
        """
        It writes the object to a file.
        
        Args:
            obj: The object to be serialized.
             path (str): The path to the file to write to.
            mode (str): The mode in which the file is opened. Defaults to "w".
        """
        with open(path, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


@FILE_HANDLERS.register(name="json")
class JsonHandler(BaseFileHandler):
    """JSON file handler."""
    
    @staticmethod
    def set_default(obj):
        """
        If the object is a set, range, numpy array, or numpy generic, convert
        it to a list. Otherwise, raise an error.
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A list of the set, range, ndarray, or generic object.
        """
        if isinstance(obj, (set, range)):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"{type(obj)} is not supported for json dump.")
    
    def load_from_fileobj(self, path: Path, **kwargs) -> str | dict | None:
        """
        This function loads a json file from a file object and returns a
        string, dictionary, or None.
        
        Args:
            path (Path): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        return json.load(path)

    def dump_to_fileobj(self, obj, path: Path, **kwargs):
        """
        It dumps the object to a file object.
        
        Args:
            obj: The object to be serialized.
            path (Path): The path to the file to write to.
        """
        kwargs.setdefault("default", self.set_default)
        json.dump(obj, path, **kwargs)

    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It takes an object and returns a string representation of that object
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A string.
        """
        kwargs.setdefault("default", self.set_default)
        return json.dumps(obj, **kwargs)


@FILE_HANDLERS.register(name="pickle")
@FILE_HANDLERS.register(name="pkl")
class PickleHandler(BaseFileHandler):
    """
    Pickle file handler.
    """
    
    def load_from_fileobj(self, path: Path, **kwargs) -> str | dict | None:
        """
        This function loads a pickle file from a file object.
        
        Args:
            path (Path): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        return pickle.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Path, **kwargs):
        """
        Takes a Python object, a path to a file, and a set of keyword arguments,
        and writes the object to the file using the pickle module.
        
        Args:
            obj: The object to be pickled.
            path (Path): The path to the file to be opened.
        """
        kwargs.setdefault("protocol", 4)
        pickle.dump(obj, path, **kwargs)
        
    def dump_to_str(self, obj, **kwargs) -> bytes:
        """
        It takes an object and returns a string representation of that object.
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A bytes object
        """
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)
        
    def load_from_file(self, file: Path, **kwargs) -> str | dict | None:
        """
        Loads a file from the file system and returns the contents as a string,
        dictionary, or None.
        
        Args:
            file (Path): Path: The file to load from.
        
        Returns:
            The return value is a string or a dictionary.
        """
        return super().load_from_file(file, mode="rb", **kwargs)
    
    def dump_to_file(self, obj, path: Path, **kwargs):
        """
        It dumps the object to a file.
        
        Args:
            obj: The object to be serialized.
            path (Path): The path to the file to which the object is to be
                dumped.
        """
        super().dump_to_file(obj, path, mode="wb", **kwargs)


@FILE_HANDLERS.register(name="xml")
class XmlHandler(BaseFileHandler):
    """
    XML file handler.
    """
    
    def load_from_fileobj(self, path: Path, **kwargs) -> str | dict | None:
        """
        It takes a path to a file, reads the file, parses the XML, and returns a
        dictionary.
        
        Args:
            path (Path): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        doc = xmltodict.parse(path.read())
        return doc

    def dump_to_fileobj(self, obj, path: Path, **kwargs):
        """
        It takes a dictionary, converts it to XML, and writes it to a file.
        
        Args:
            obj: The object to be dumped.
            path (Path): The path to the file to be read.
        """
        assert_dict(obj)
        with open(path, "w") as path:
            path.write(xmltodict.unparse(obj, pretty=True))
        
    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It takes a dictionary, converts it to XML, and returns the XML as a
        string.
        
        Args:
            obj: The object to be serialized.
        
        Returns:
            A string.
        """
        assert_dict(obj)
        return xmltodict.unparse(obj, pretty=True)


@FILE_HANDLERS.register(name="yaml")
@FILE_HANDLERS.register(name="yml")
class YamlHandler(BaseFileHandler):
    """
    YAML file handler.
    """
    
    def load_from_fileobj(self, path: Path, **kwargs) -> str | dict | None:
        """
        It loads a YAML file from a file object.
        
        Args:
            path (Path): The path to the file to load.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        kwargs.setdefault("Loader", FullLoader)
        return yaml.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Path, **kwargs):
        """
        It takes a Python object, a path to a file, and a set of keyword
        arguments, and writes the object to the file using the `Dumper` class.
        
        Args:
            obj: The Python object to be serialized.
            path (Path): The file object to dump to.
        """
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, path, **kwargs)

    def dump_to_str(self, obj, **kwargs) -> str:
        """
        It dumps the object to a string.
        
        Args:
            obj: the object to be serialized.
        
        Returns:
            A string.
        """
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
