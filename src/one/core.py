#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core module that defines all basic types, custom types, classes, helper
functions, constants, and globals used throughout `One` package.

Basically, this is just a one glorified utils package that reduce the
complexity of import statements.

Taxonomy:
    |
    |__ Assertion
    |__ Constant / Global
    |__ Conversion
    |__ Dataclass
    |__ Dataset
    |     |__ Unlabeled
    |     |__ Labeled
    |     |__ Classification
    |     |__ Detection
    |     |__ Enhancement
    |     |__ Segmentation
    |     |__ Multitask
    |__ Device
    |__ Factory
    |__ File
    |__ Logging
    |__ Serialization
    |__ Transform
    |__ Typing
    |__ All
"""

from __future__ import annotations

import collections
import functools
import glob
import inspect
import itertools
import json
import logging
import os
import pathlib
import pickle
import shutil
import sys
import time
import types
import typing
import uuid
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from collections import abc
from collections import OrderedDict
from copy import copy
from copy import deepcopy
from enum import Enum
from numbers import Number
from pathlib import Path
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Sequence
from typing import TypeVar
from typing import Union

import cv2
import numpy as np
import PIL
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import validators
import xmltodict
import yaml
from joblib import delayed
from joblib import Parallel
from matplotlib import pyplot as plt
from multipledispatch import dispatch
from munch import Munch
from ordered_enum import OrderedEnum
from PIL import Image
from pynvml import nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetMemoryInfo
from pynvml import nvmlInit
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import SpinnerColumn
from rich.progress import Task
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn
from rich.table import Column
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric

try:
    from yaml import CLoader as FullLoader, CDumper as Dumper
except ImportError:
    from yaml import FullLoader, Dumper


# H1: - Assertion --------------------------------------------------------------

def is_basename(path: Path_ | None) -> bool:
    """
    If the path is not None, and the parent of the path is the current
    directory, then the path is a basename.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return str(path) == path.name


def is_bmp_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is `.bmp`, then return True.
    Otherwise, return False.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() == ImageFormat.BMP.value


def is_ckpt_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is `.ckpt`, then return True.
    Otherwise, return False.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".ckpt"]
 

def is_class(input: Any) -> bool:
    return inspect.isclass(input)


def is_collection(input: Any) -> bool:
    return isinstance(input, Collection)
    

def is_dict(input: Any) -> bool:
    return isinstance(input, (dict, Munch))


def is_dict_contain_key(input: Any, key: Any) -> bool:
    assert_dict(input)
    if key in input:
        return True
    else:
        raise ValueError(f"`input` dict must contain the key `{key}`.")


def is_dict_of(input: dict, item_type: type) -> bool:
    """Check whether `input` is a dictionary of some type.
    
    Args:
        input (dict):
            Dictionary to be checked.
        item_type (type):
            Expected type of items.
    
    Return:
        (bool):
            `True` if `s` is a dictionary containing item of type `item_type`.
            Else `False`.
    """
    assert_valid_type(item_type)
    return all(isinstance(v, item_type) for k, v in input.items())


def is_dir(path: Path_ | None) -> bool:
    """
    If the path is a directory, then return True.
    
    Args:
        path (Path_ | None): The path to the directory.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_dir()
    

def is_float(input: Any) -> bool:
    return isinstance(input, float)


def is_image_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is in the list of image formats,
    then return True.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in ImageFormat.values()


def is_int(input: Any) -> bool:
    return isinstance(input, int)


def is_iterable(input: Any) -> bool:
    return isinstance(input, Iterable)


def is_json_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is json, return True.
    Otherwise, return False.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".json"]


def is_list(input: Any) -> bool:
    return isinstance(input, list)


def is_list_of(input: Any, item_type: type) -> bool:
    """Check whether `l` is a list of some type.

    Args:
        input (list):
            List to be checked.
        item_type (type):
            Expected type of items.

    Return:
        (bool):
            `True` if `s` is a list containing item of type `item_type`.
            Else `False`.
    """
    return is_sequence_of(input=input, item_type=item_type, seq_type=list)


def is_name(path: Path_ | None) -> bool:
    """
    If the path is None, return False. If the path is the same as the path's
    stem, return True. Otherwise, return False.

    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path == path.stem


def is_negative_number(input: Any) -> bool:
    assert_number(input)
    if input < 0.0:
        return True
    return False
    
    
def is_number(input: Any) -> bool:
    return isinstance(input, Number)


def is_number_divisible_to(input: Any, k: int) -> bool:
    assert_number(input)
    assert_number(k)
    return input % k == 0


def is_number_in_range(input: Any, start: Number, end: Number) -> bool:
    return start <= input <= end


def is_numpy(input: Any) -> bool:
    return isinstance(input, np.ndarray)


def is_numpy_of_atleast_ndim(input: Any, ndim: int) -> bool:
    assert_numpy(input)
    assert_int(ndim)
    return input.ndim >= ndim
    

def is_numpy_of_channels(input: Any, channels: Ints) -> bool:
    from one.vision.transformation import get_num_channels
    assert_numpy_of_atleast_ndim(input, 3)
    channels = to_list(channels)
    assert_list(channels)
    c = get_num_channels(input)
    return c in channels


def is_numpy_of_ndim(input: Any, ndim: int) -> bool:
    assert_numpy(input=input)
    assert_int(input=ndim)
    return input.ndim == ndim
    

def is_numpy_of_ndim_in_range(input: Any, start: int, end: int) -> bool:
    assert_numpy(input)
    assert_int(start)
    assert_int(end)
    return start <= input.ndim <= end


def is_positive_number(input: Any) -> bool:
    assert_number(input)
    return input >= 0.0
    

def is_same_length(input1: Sequence, input2: Sequence) -> bool:
    assert_sequence(input1)
    assert_sequence(input2)
    return len(input1) == len(input2)


def is_same_shape(
    input1: Tensor | np.ndarray, input2: Tensor | np.ndarray
) -> bool:
    return input1.shape == input2.shape


def is_sequence(input: Any) -> bool:
    return isinstance(input, Sequence)


def is_sequence_of(input: Sequence, item_type: type, seq_type: Any = None) -> bool:
    """Check whether `s` is a sequence of some type.
    
    Args:
        input (Sequence):
            Sequence to be checked.
        item_type (type):
            Expected type of sequence items.
        seq_type (type, None):
            Expected sequence type. Default: `None`.
    
    Return:
        (bool):
            `True` if `s` is a sequence of type `seq_type` containing item of
            type `item_type`. Else `False`.
    """
    if seq_type is None:
        seq_type = Sequence
    assert_valid_type(seq_type)
    
    if not isinstance(input, seq_type):
        return False
    for item in input:
        if not isinstance(item, item_type):
            return False
    return True


def is_sequence_of_length(input: Any, length: int) -> bool:
    assert_sequence(input)
    assert_int(length)
    return len(input) == length


def is_stem(path: Path_ | None) -> bool:
    """
    If the path is not None, and the parent of the path is the current
    directory, and the path has no extension, then the path is a stem.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return str(path) == path.stem


def is_str(input: Any) -> bool:
    return isinstance(input, str)


def is_tensor(input: Any) -> bool:
    return isinstance(input, Tensor)


def is_tensor_of_atleast_ndim(input: Any, ndim: int) -> bool:
    assert_tensor(input)
    assert_int(ndim)
    return input.ndim >= ndim
    

def is_tensor_of_channels(input: Any, channels: Ints) -> bool:
    from one.vision.transformation import get_num_channels
    assert_tensor_of_atleast_ndim(input, 3)
    channels = to_list(channels)
    assert_list(channels)
    c = get_num_channels(input)
    return c in channels


def is_tensor_of_ndim(input: Any, ndim: int) -> bool:
    assert_tensor(input)
    assert_int(ndim)
    return input.ndim == ndim
    

def is_tensor_of_ndim_in_range(input: Any, start: int, end: int) -> bool:
    assert_tensor(input)
    assert_int(start)
    assert_int(end)
    return start <= input.ndim <= end
    

def is_torch_saved_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is one of the following:
        - pt
        - pth
        - weights
        - ckpt
    Then return True. Otherwise, return False.
    
    Args:
        path (Path_ | None): The path to the file to be checked.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() \
           and path.suffix.lower() in [".pt", ".pth", ".weights", ".ckpt"]


def is_tuple(input: Any) -> bool:
    return isinstance(input, tuple)


def is_tuple_of(input: Any, item_type: type) -> bool:
    """Check whether `input` is a tuple of some type.
    
    Args:
        input (tuple):
            Dictionary to be checked.
        item_type (type):
            Expected type of items.
    
    Return:
        (bool):
            `True` if `s` is a tuple containing item of type `item_type`.
            Else `False`.
    """
    return is_sequence_of(input=input, item_type=item_type, seq_type=tuple)


def is_valid_type(input: Any) -> bool:
    return isinstance(input, type)


def is_url(path: Path_ | None) -> bool:
    """
    If the path is a URL, return True, otherwise return False.
    
    Args:
        path (Path_ | None): The path to the file or directory.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    return not isinstance(validators.url(path), validators.ValidationFailure)


def is_url_or_file(path: Path_ | None) -> bool:
    """
    If the path is a URL or a file, return True. Otherwise, return False
    
    Args:
        path (Path_ | None): The path to the file or URL.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() or \
           not isinstance(validators.url(path), validators.ValidationFailure)


def is_txt_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is txt, return True, otherwise
    return False.

    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".txt"]


def is_value_in_collection(input: Any, collection: Any) -> bool:
    assert_collection(collection)
    return input in collection


def is_video_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is in the list of video
    formats, then return True.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in VideoFormat.values()


def is_video_stream(path: Path_ | None) -> bool:
    """
    If the path is not None and contains the string 'rtsp', return True,
    otherwise return False.
    
    Args:
        path (Path_ | None): The path to the video file or stream.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    return "rtsp" in path.lower()


def is_weights_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is `pt` or `pth`, then return
    True. Otherwise, return False.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".pt", ".pth"]


def is_xml_file(path: Path_ | None) -> bool:
    """
    If the path is a file and the file extension is xml, return True, otherwise
    return False.
    
    Args:
        path (Path_ | None): The path to the file.
    
    Returns:
        A boolean value.
    """
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".xml"]


def is_yaml_file(path: Path_ | None) -> bool:
    """Check if the given path is a `.yaml` file."""
    if path is None:
        return False
    path = Path(path)
    return path.is_file() and path.suffix.lower() in [".yaml", ".yml"]


def assert_bmp_file(path: Path_ | None):
    if not is_bmp_file(path):
        raise ValueError()


def assert_basename(path: Path_ | None) -> bool:
    if not is_basename(path):
        raise ValueError()


def assert_ckpt_file(path: Path_ | None):
    if not is_ckpt_file(path):
        raise ValueError()
    
    
def assert_class(input: Any):
    if not is_class(input):
        raise TypeError(
            f"`input` must be a class type. But got: {type(input)}."
        )


def assert_collection(input: Any):
    if not is_collection(input):
        raise TypeError(
            f"`input` must be a `Collection`. But got: {type(input)}."
        )


def assert_dict(input: Any):
    if not is_dict(input):
        raise TypeError(f"`input` must be a `dict`. But got: {type(input)}.")


def assert_dict_contain_key(input: Any, key: Any):
    if not is_dict_contain_key(input, key):
        raise ValueError(f"`input` dict must contain the key `{key}`.")


def assert_dict_of(input: dict, item_type: type):
    if not is_dict_of(input, item_type):
        raise TypeError()


def assert_dir(path: Path_ | None):
    if not is_dir(path):
        raise ValueError()


def assert_float(input: Any):
    if not is_float(input):
        raise TypeError(f"`input` must be a `float`. But got: {type(input)}.")


def assert_image_file(path: Path_ | None):
    if not is_image_file(path):
        raise ValueError()
    
    
def assert_int(input: Any):
    if not is_int(input):
        raise TypeError(f"`input` must be an `int`. But got: {type(input)}.")


def assert_iterable(input: Any):
    if not is_iterable(input):
        raise TypeError(
            f"`inputs` must be an iterable object. But got: {type(input)}."
        )


def assert_json_file(path: Path_ | None):
    if not is_json_file(path):
        raise ValueError()


def assert_list(input: Any):
    if not is_list(input):
        raise TypeError(f"`input` must be a `list`. But got: {type(input)}.")


def assert_list_of(input: Any, item_type: type) -> bool:
    if not is_list_of(input, item_type):
        raise TypeError()


def assert_name(path: Path_ | None) -> bool:
    if not is_name(path):
        raise ValueError()


def assert_negative_number(input: Any):
    if not is_negative_number(input):
        raise ValueError(f"`input` must be negative. But got: {input}.")


def assert_number(input: Any):
    if not is_number(input=input):
        raise TypeError(f"`input` must be a `Number`. But got: {type(input)}.")
    

def assert_number_divisible_to(input: Any, k: int):
    if not is_number_divisible_to(input, k):
        raise ValueError(
            f"`input` must be divisible by `{k}`. "
            f"But got: {input} % {k} != 0."
        )


def assert_number_in_range(input: Any, start: Number, end: Number):
    if not is_number_in_range(input, start, end):
        raise ValueError(
            f"Require {start} <= `input` <= {end}. But got: {input}."
        )


def assert_numpy(input: Any) -> bool:
    if not is_numpy(input):
        raise TypeError(
            f"`input` must be a `np.ndarray`. But got: {type(input)}."
        )
        

def assert_numpy_of_atleast_ndim(input: Any, ndim: int):
    if not is_numpy_of_atleast_ndim(input, ndim):
        raise TypeError(
            f"`input` must be a `np.ndarray` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )


def assert_numpy_of_channels(input: Any, channels: Ints):
    if not is_numpy_of_channels(input, channels):
        raise TypeError(
            f"`input` must be a `np.ndarray` of channels `{channels}`."
        )


def assert_numpy_of_ndim(input: Any, ndim: int):
    if not is_numpy_of_ndim(input, ndim):
        raise ValueError(
            f"Require `input.ndim` == {ndim}. But got: {input.ndim}."
        )


def assert_numpy_of_ndim_in_range(input: Any, start: int, end: int):
    if not is_numpy_of_ndim_in_range(input, start, end):
        raise ValueError(
            f"Require {start} <= `input.ndim` <= {end}. But got: {input.ndim}."
        )


def assert_positive_number(input: Any):
    if not is_positive_number(input):
        raise ValueError(f"`input` must be positive. But got: {input}.")
    

def assert_same_length(input1: Sequence, input2: Sequence) -> bool:
    if not is_same_length(input1, input2):
        raise ValueError(
            f"`input1` and `input2` must have the same length. "
            f"But got: {len(input1)} != {len(input2)}."
        )
    
    
def assert_same_shape(input1: Tensor | np.ndarray, input2: Tensor | np.ndarray):
    if not is_same_shape(input1, input2):
        raise ValueError(
            f"`input1` and `input2` must have the same shape. "
            f"But got: {input1.shape} != {input2.shape}."
        )
  
    
def assert_sequence(input: Any) -> bool:
    if not is_sequence(input):
        raise TypeError(
            f"`input` must be a `Sequence`. But got: {type(input)}."
        )


def assert_sequence_of(input: Sequence, item_type: type, seq_type: Any = None):
    if not is_sequence_of(input, item_type, seq_type):
        raise TypeError()


def assert_sequence_of_length(input: Any, length: int):
    if not is_sequence_of_length(input, length):
        raise TypeError(
            f"`input` must be a sequence of length `{length}`. "
            f"But got: {len(input)}."
        )
    

def assert_stem(path: Path_ | None) -> bool:
    if not is_stem(path):
        raise ValueError()

    
def assert_str(input: Any):
    if not is_str(input):
        raise TypeError(f"`input` must be a `str`. But got: {type(input)}.")


def assert_tensor(input: Any):
    if not is_tensor(input):
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    

def assert_tensor_of_atleast_ndim(input: Any, ndim: int):
    if not is_tensor_of_atleast_ndim(input, ndim):
        raise TypeError(
            f"`input` must be a `Tensor` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )


def assert_tensor_of_channels(input: Any, channels: Ints):
    if not is_tensor_of_channels(input, channels):
        raise TypeError(f"`input` must be a `Tensor` of channels `{channels}`. ")
    

def assert_tensor_of_ndim(input: Any, ndim: int):
    if not is_tensor_of_ndim(input, ndim):
        raise TypeError(
            f"`input` must be a `Tensor` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )
 
 
def assert_tensor_of_ndim_in_range(input: Any, start: int, end: int):
    if not is_tensor_of_ndim_in_range(input, start, end):
        raise ValueError(
            f"Require {start} <= `input.ndim` <= {end}. But got: {input.ndim}."
        )
    

def assert_torch_saved_file(path: Path_ | None) -> bool:
    if not is_torch_saved_file(path):
        raise ValueError()
    
    
def assert_tuple(input: Any):
    if not is_tuple(input):
        raise TypeError(f"`input` must be a `tuple`. But got: {type(input)}.")
    
    
def assert_tuple_of(input: Any, item_type: type):
    if not is_tuple_of(input, item_type):
        raise TypeError()


def assert_txt_file(path: Path_ | None) -> bool:
    if not is_txt_file(path):
        raise ValueError()


def assert_url(path: Path_ | None) -> bool:
    if not is_url(path):
        raise ValueError()


def assert_url_or_file(path: Path_ | None) -> bool:
    if not is_url_or_file(path):
        raise ValueError()


def assert_valid_type(input: Any):
    if not is_valid_type(input):
        raise TypeError(f"`input` must be a valid type. But got: {input}.")


def assert_value_in_collection(input: Any, collection: Any):
    if not is_value_in_collection(input, collection):
        raise ValueError(
            f"`input` must be included in `collection`. "
            f"But got: {input} not in {collection}."
        )
    
    
def assert_video_file(path: Path_ | None) -> bool:
    if not is_video_file(path):
        raise ValueError()


def assert_video_stream(path: Path_ | None) -> bool:
    if not is_video_stream(path):
        raise ValueError()


def assert_weights_file(path: Path_ | None) -> bool:
    if not is_weights_file(path):
        raise ValueError()


def assert_xml_file(path: Path_ | None) -> bool:
    if not is_xml_file(path):
        raise ValueError()


def assert_yaml_file(path: Path_ | None) -> bool:
    if not is_yaml_file(path):
        raise ValueError()
    
    
# H1: - Constant / Global ------------------------------------------------------

# H2: - Enum -------------------------------------------------------------------

class AppleRGB(OrderedEnum):
    """
    Define 12 Apple colors.
    """
    GRAY   = (128, 128, 128)
    RED    = (255, 59 , 48 )
    GREEN  = ( 52, 199, 89 )
    BLUE   = (  0, 122, 255)
    ORANGE = (255, 149, 5  )
    YELLOW = (255, 204, 0  )
    BROWN  = (162, 132, 94 )
    PINK   = (255, 45 , 85 )
    PURPLE = ( 88, 86 , 214)
    TEAL   = ( 90, 200, 250)
    INDIGO = ( 85, 190, 240)
    BLACK  = (  0, 0  , 0  )
    WHITE  = (255, 255, 255)
    
    @staticmethod
    def values() -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in AppleRGB]


class BasicRGB(OrderedEnum):
    """
    Define 12 basic colors.
    """
    BLACK   = (0  , 0  , 0  )
    WHITE   = (255, 255, 255)
    RED     = (255, 0  , 0  )
    LIME    = (0  , 255, 0  )
    BLUE    = (0  , 0  , 255)
    YELLOW  = (255, 255, 0  )
    CYAN    = (0  , 255, 255)
    MAGENTA = (255, 0  , 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    MAROON  = (128, 0  , 0  )
    OLIVE   = (128, 128, 0  )
    GREEN   = (0  , 128, 0  )
    PURPLE  = (128, 0  , 128)
    TEAL    = (0  , 128, 128)
    NAVY    = (0  , 0  , 128)
    
    @staticmethod
    def values() -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in BasicRGB]


class BBoxFormat(Enum):
    CXCYAR      = "cxcyar"
    CXCYRH      = "cxcyrh"
    CXCYWH      = "cxcywh"
    CXCYWH_NORM = "cxcywh_norm"
    XYXY        = "xyxy"
    XYWH        = "xywh"
    
    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "cxcyar"     : BBoxFormat.CXCYAR,
            "cxcyrh"     : BBoxFormat.CXCYRH,
            "cxcywh"     : BBoxFormat.CXCYWH,
            "cxcywh_norm": BBoxFormat.CXCYWH_NORM,
            "xyxy"       : BBoxFormat.XYXY,
            "xywh"       : BBoxFormat.XYWH,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: BBoxFormat.CXCYAR,
            1: BBoxFormat.CXCYRH,
            2: BBoxFormat.CXCYWH,
            3: BBoxFormat.CXCYWH_NORM,
            4: BBoxFormat.XYXY,
            5: BBoxFormat.XYWH,
        }
    
    @staticmethod
    def from_str(value: str) -> BBoxFormat:
        """
        It takes a string and returns an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(BBoxFormat.str_mapping, value.lower())
        return BBoxFormat.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> BBoxFormat:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(BBoxFormat.int_mapping, value)
        return BBoxFormat.int_mapping()[value]

    @staticmethod
    def from_value(value: Any) -> BBoxFormat | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, BBoxFormat):
            return value
        if isinstance(value, str):
            return BBoxFormat.from_str(value)
        if isinstance(value, int):
            return BBoxFormat.from_int(value)
        error_console.log(
            f"`value` must be `BBoxFormat`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
    
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [b for b in BBoxFormat]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [b.value for b in BBoxFormat]


class CFA(Enum):
    """
    Define the configuration of the color filter array.

    So far only bayer images is supported and the enum sets the pixel order for
    bayer. Note that this can change due to things like rotations and cropping
    of images. Take care if including the translations in pipeline. This
    implementations is optimized to be reasonably fast, look better than simple
    nearest neighbour. On top of this care is taken to make it reversible going
    raw -> rgb -> raw. the raw samples remain intact during conversion and only
    unknown samples are interpolated.

    Names are based on the OpenCV convention where the BG indicates pixel
    1,1 (counting from 0,0) is blue and its neighbour to the right is green.
    In that case the top left pixel is red. Other options are GB, RG and GR

    Reference:
        https://en.wikipedia.org/wiki/Color_filter_array
    """
    BG = 0
    GB = 1
    RG = 2
    GR = 3


class DistanceMetric(Enum):
    BRAYCURTIS         = "braycurtis"
    CANBERRA           = "canberra"
    CHEBYSHEV          = "chebyshev"
    CITYBLOCK          = "cityblock"
    CORRELATION        = "correlation"
    COSINE             = "cosine"
    DICE               = "dice"
    DIRECTED_HAUSDORFF = "directed_hausdorff"
    EUCLIDEAN          = "euclidean"
    HAMMING            = "hamming"
    JACCARD            = "jaccard"
    JENSENSHANNON      = "jensenshannon"
    KULCZYNSKI1        = "kulczynski1"
    KULSINSKI          = "kulsinski"
    MAHALANOBIS        = "mahalanobis"
    MINKOWSKI          = "minkowski"
    ROGERSTANIMOTO     = "rogerstanimoto"
    RUSSELLRAO         = "russellrao"
    SEUCLIDEAN         = "seuclidean"
    SOKALMICHENER      = "sokalmichener"
    SOKALSNEATH        = "sokalsneath"
    SQEUCLIDEAN        = "sqeuclidean"
    YULE               = "yule"
    
    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "braycurtis"        : DistanceMetric.BRAYCURTIS,
            "canberra"          : DistanceMetric.CANBERRA,
            "chebyshev"         : DistanceMetric.CHEBYSHEV,
            "cityblock"         : DistanceMetric.CITYBLOCK,
            "correlation"       : DistanceMetric.CORRELATION,
            "cosine"            : DistanceMetric.COSINE,
            "dice"              : DistanceMetric.DICE,
            "directed_hausdorff": DistanceMetric.DIRECTED_HAUSDORFF,
            "euclidean"         : DistanceMetric.EUCLIDEAN,
            "hamming"           : DistanceMetric.HAMMING,
            "jaccard"           : DistanceMetric.JACCARD,
            "jensenshannon"     : DistanceMetric.JENSENSHANNON,
            "kulczynski1"       : DistanceMetric.KULCZYNSKI1,
            "kulsinski"         : DistanceMetric.KULSINSKI,
            "mahalanobis"       : DistanceMetric.MAHALANOBIS,
            "minkowski"         : DistanceMetric.MINKOWSKI,
            "rogerstanimoto"    : DistanceMetric.ROGERSTANIMOTO,
            "russellrao"        : DistanceMetric.RUSSELLRAO,
            "seuclidean"        : DistanceMetric.SEUCLIDEAN,
            "sokalmichener"     : DistanceMetric.SOKALMICHENER,
            "sokalsneath"       : DistanceMetric.SOKALSNEATH,
            "sqeuclidean"       : DistanceMetric.SQEUCLIDEAN,
            "yule"              : DistanceMetric.YULE,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : DistanceMetric.BRAYCURTIS,
            1 : DistanceMetric.CANBERRA,
            2 : DistanceMetric.CHEBYSHEV,
            3 : DistanceMetric.CITYBLOCK,
            4 : DistanceMetric.CORRELATION,
            5 : DistanceMetric.COSINE,
            6 : DistanceMetric.DICE,
            7 : DistanceMetric.DIRECTED_HAUSDORFF,
            8 : DistanceMetric.EUCLIDEAN,
            9 : DistanceMetric.HAMMING,
            10: DistanceMetric.JACCARD,
            11: DistanceMetric.JENSENSHANNON,
            12: DistanceMetric.KULCZYNSKI1,
            13: DistanceMetric.KULSINSKI,
            14: DistanceMetric.MAHALANOBIS,
            15: DistanceMetric.MINKOWSKI,
            16: DistanceMetric.ROGERSTANIMOTO,
            17: DistanceMetric.RUSSELLRAO,
            18: DistanceMetric.SEUCLIDEAN,
            19: DistanceMetric.SOKALMICHENER,
            20: DistanceMetric.SOKALSNEATH,
            21: DistanceMetric.SQEUCLIDEAN,
            22: DistanceMetric.YULE,
        }
    
    @staticmethod
    def from_str(value: str) -> DistanceMetric:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(DistanceMetric.str_mapping, value.lower())
        return DistanceMetric.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> DistanceMetric:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(DistanceMetric.int_mapping, value)
        return DistanceMetric.int_mapping()[value]
    
    @staticmethod
    def from_value(value: Any) -> DistanceMetric | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, DistanceMetric):
            return value
        if isinstance(value, str):
            return DistanceMetric.from_str(value)
        if isinstance(value, int):
            return DistanceMetric.from_int(value)
        error_console.log(
            f"`value` must be `DistanceMetric`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
    
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in DistanceMetric]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in DistanceMetric]
 

class ImageFormat(Enum):
    BMP  = ".bmp"
    DNG	 = ".dng"
    JPG  = ".jpg"
    JPEG = ".jpeg"
    PNG  = ".png"
    PPM  = ".ppm"
    TIF  = ".tif"
    TIFF = ".tiff"
    
    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "bmp" : ImageFormat.BMP,
            "dng" : ImageFormat.DNG,
            "jpg" : ImageFormat.JPG,
            "jpeg": ImageFormat.JPEG,
            "png" : ImageFormat.PNG,
            "ppm" : ImageFormat.PPM,
            "tif" : ImageFormat.TIF,
            "tiff": ImageFormat.TIF,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: ImageFormat.BMP,
            1: ImageFormat.DNG,
            2: ImageFormat.JPG,
            3: ImageFormat.JPEG,
            4: ImageFormat.PNG,
            5: ImageFormat.PPM,
            6: ImageFormat.TIF,
            7: ImageFormat.TIF,
        }
    
    @staticmethod
    def from_str(value: str) -> ImageFormat:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(ImageFormat.str_mapping, value.lower())
        return ImageFormat.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> ImageFormat:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(ImageFormat.int_mapping, value)
        return ImageFormat.int_mapping()[value]

    @staticmethod
    def from_value(value: Enum_) -> ImageFormat | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Enum_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, ImageFormat):
            return value
        if isinstance(value, str):
            return ImageFormat.from_str(value)
        if isinstance(value, int):
            return ImageFormat.from_int(value)
        error_console.log(
            f"`value` must be `ImageFormat`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
    
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in ImageFormat]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in ImageFormat]


class InterpolationMode(Enum):
    BICUBIC       = "bicubic"
    BILINEAR      = "bilinear"
    NEAREST       = "nearest"
    # For PIL compatibility
    BOX           = "box"
    HAMMING       = "hamming"
    LANCZOS       = "lanczos"
    # For opencv compatibility
    AREA          = "area"
    CUBIC         = "cubic"
    LANCZOS4      = "lanczos4"
    LINEAR        = "linear"
    LINEAR_EXACT  = "linear_exact"
    MAX           = "max"
    NEAREST_EXACT = "nearest_exact"

    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "bicubic"      : InterpolationMode.BICUBIC,
            "bilinear"     : InterpolationMode.BILINEAR,
            "nearest"      : InterpolationMode.NEAREST,
            "box"          : InterpolationMode.BOX,
            "hamming"      : InterpolationMode.HAMMING,
            "lanczos"      : InterpolationMode.LANCZOS,
            "area"         : InterpolationMode.AREA,
            "cubic"        : InterpolationMode.CUBIC,
            "lanczos4"     : InterpolationMode.LANCZOS4,
            "linear"       : InterpolationMode.LINEAR,
            "linear_exact" : InterpolationMode.LINEAR_EXACT,
            "max"          : InterpolationMode.MAX,
            "nearest_exact": InterpolationMode.NEAREST_EXACT,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : InterpolationMode.BICUBIC,
            1 : InterpolationMode.BILINEAR,
            2 : InterpolationMode.NEAREST,
            3 : InterpolationMode.BOX,
            4 : InterpolationMode.HAMMING,
            5 : InterpolationMode.LANCZOS,
            6 : InterpolationMode.AREA,
            7 : InterpolationMode.CUBIC,
            8 : InterpolationMode.LANCZOS4,
            9 : InterpolationMode.LINEAR,
            10: InterpolationMode.LINEAR_EXACT,
            11: InterpolationMode.MAX,
            12: InterpolationMode.NEAREST_EXACT,
        }

    @staticmethod
    def cv_modes_mapping() -> dict:
        """
        It maps the `InterpolationMode` enum to the corresponding OpenCV
        interpolation mode.
        
        Returns:
            A dictionary of the different interpolation modes.
        """
        return {
            InterpolationMode.AREA    : cv2.INTER_AREA,
            InterpolationMode.CUBIC   : cv2.INTER_CUBIC,
            InterpolationMode.LANCZOS4: cv2.INTER_LANCZOS4,
            InterpolationMode.LINEAR  : cv2.INTER_LINEAR,
            InterpolationMode.MAX     : cv2.INTER_MAX,
            InterpolationMode.NEAREST : cv2.INTER_NEAREST,
        }

    @staticmethod
    def pil_modes_mapping() -> dict:
        """
        It maps the `InterpolationMode` enum to the corresponding PIL
        interpolation mode.
        
        Returns:
            A dictionary with the keys being the InterpolationMode enum and the
            values being the corresponding PIL interpolation mode.
        """
        return {
            InterpolationMode.NEAREST : 0,
            InterpolationMode.LANCZOS : 1,
            InterpolationMode.BILINEAR: 2,
            InterpolationMode.BICUBIC : 3,
            InterpolationMode.BOX     : 4,
            InterpolationMode.HAMMING : 5,
        }
    
    @staticmethod
    def from_str(value: str) -> InterpolationMode:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(InterpolationMode.str_mapping, value.lower())
        return InterpolationMode.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> InterpolationMode:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(InterpolationMode.int_mapping, value)
        return InterpolationMode.int_mapping()[value]

    @staticmethod
    def from_value(value: Enum_) -> InterpolationMode | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Enum_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, InterpolationMode):
            return value
        if isinstance(value, str):
            return InterpolationMode.from_str(value)
        if isinstance(value, int):
            return InterpolationMode.from_int(value)
        error_console.log(
            f"`value` must be `InterpolationMode`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
        
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in InterpolationMode]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in InterpolationMode]


class MemoryUnit(Enum):
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"

    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "b" : MemoryUnit.B,
            "kb": MemoryUnit.KB,
            "mb": MemoryUnit.MB,
            "gb": MemoryUnit.GB,
            "tb": MemoryUnit.TB,
            "pb": MemoryUnit.PB,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: MemoryUnit.B,
            1: MemoryUnit.KB,
            2: MemoryUnit.MB,
            3: MemoryUnit.GB,
            4: MemoryUnit.TB,
            5: MemoryUnit.PB,
        }
    
    @staticmethod
    def byte_conversion_mapping():
        """
        It returns a dictionary that maps the MemoryUnit enum to the number of
        bytes in that unit.
        
        Returns:
            A dictionary with the keys being the MemoryUnit enum and the values
            being the number of bytes in each unit.
        """
        return {
            MemoryUnit.B : 1024 ** 0,
            MemoryUnit.KB: 1024 ** 1,
            MemoryUnit.MB: 1024 ** 2,
            MemoryUnit.GB: 1024 ** 3,
            MemoryUnit.TB: 1024 ** 4,
            MemoryUnit.PB: 1024 ** 5,
        }
    
    @staticmethod
    def from_str(value: str) -> MemoryUnit:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(MemoryUnit.str_mapping, value.lower())
        return MemoryUnit.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> MemoryUnit:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(MemoryUnit.int_mapping, value)
        return MemoryUnit.int_mapping()[value]
    
    @staticmethod
    def from_value(value: Any) -> MemoryUnit | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, MemoryUnit):
            return value
        if isinstance(value, str):
            return MemoryUnit.from_str(value)
        if isinstance(value, int):
            return MemoryUnit.from_int(value)
        error_console.log(
            f"`value` must be `MemoryUnit`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
    
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in MemoryUnit]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in MemoryUnit]


class ModelPhase(Enum):
    TRAINING  = "training"
    # Produce predictions, calculate losses and metrics, update weights at
    # the end of each epoch/step.
    TESTING   = "testing"
    # Produce predictions, calculate losses and metrics,
    # DO NOT update weights at the end of each epoch/step.
    INFERENCE = "inference"
    # Produce predictions ONLY.
    
    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "training" : ModelPhase.TRAINING,
            "testing"  : ModelPhase.TESTING,
            "inference": ModelPhase.INFERENCE,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: ModelPhase.TRAINING,
            1: ModelPhase.TESTING,
            2: ModelPhase.INFERENCE,
        }

    @staticmethod
    def from_str(value: str) -> ModelPhase:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(ModelPhase.str_mapping, value.lower())
        return ModelPhase.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> ModelPhase:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(ModelPhase.int_mapping, value)
        return ModelPhase.int_mapping()[value]

    @staticmethod
    def from_value(value: Any) -> ModelPhase | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, ModelPhase):
            return value
        if isinstance(value, str):
            return ModelPhase.from_str(value)
        if isinstance(value, int):
            return ModelPhase.from_int(value)
        error_console.log(
            f"`value` must be `ModelPhase`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None

    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in ModelPhase]

    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in ModelPhase]


class PaddingMode(Enum):
    CONSTANT      = "constant"
    # For torch compatibility
    CIRCULAR      = "circular"
    REFLECT       = "reflect"
    REPLICATE     = "replicate"
    # For numpy compatibility
    EDGE          = "edge"
    EMPTY         = "empty"
    LINEAR_RAMP   = "linear_ramp"
    MAXIMUM       = "maximum"
    MEAN          = "mean"
    MEDIAN        = "median"
    MINIMUM       = "minimum"
    SYMMETRIC     = "symmetric"
    WRAP          = "wrap"

    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "constant"   : PaddingMode.CONSTANT,
            "circular"   : PaddingMode.CIRCULAR,
            "reflect"    : PaddingMode.REFLECT,
            "replicate"  : PaddingMode.REPLICATE,
            "edge"       : PaddingMode.EDGE,
            "empty"      : PaddingMode.EMPTY,
            "linear_ramp": PaddingMode.LINEAR_RAMP,
            "maximum"    : PaddingMode.MAXIMUM,
            "mean"       : PaddingMode.MEAN,
            "median"     : PaddingMode.MEDIAN,
            "minimum"    : PaddingMode.MINIMUM,
            "symmetric"  : PaddingMode.SYMMETRIC,
            "wrap"       : PaddingMode.WRAP,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : PaddingMode.CONSTANT,
            1 : PaddingMode.CIRCULAR,
            2 : PaddingMode.REFLECT,
            3 : PaddingMode.REPLICATE,
            4 : PaddingMode.EDGE,
            5 : PaddingMode.EMPTY,
            6 : PaddingMode.LINEAR_RAMP,
            7 : PaddingMode.MAXIMUM,
            8 : PaddingMode.MEAN,
            9 : PaddingMode.MEDIAN,
            10: PaddingMode.MINIMUM,
            11: PaddingMode.SYMMETRIC,
            12: PaddingMode.WRAP,
        }

    @staticmethod
    def from_str(value: str) -> PaddingMode:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(PaddingMode.str_mapping, value)
        return PaddingMode.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> PaddingMode:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(PaddingMode.int_mapping, value)
        return PaddingMode.int_mapping()[value]

    @staticmethod
    def from_value(value: Any) -> PaddingMode | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, PaddingMode):
            return value
        if isinstance(value, str):
            return PaddingMode.from_str(value)
        if isinstance(value, int):
            return PaddingMode.from_int(value)
        error_console.log(
            f"`value` must be `PaddingMode`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
        
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in PaddingMode]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in PaddingMode]


class RGB(OrderedEnum):
    """
    Define 138 colors.
    """
    MAROON                  = (128, 0  , 0  )
    DARK_RED                = (139, 0  , 0  )
    BROWN                   = (165, 42 , 42 )
    FIREBRICK               = (178, 34 , 34 )
    CRIMSON                 = (220, 20 , 60 )
    RED                     = (255, 0  , 0  )
    TOMATO                  = (255, 99 , 71 )
    CORAL                   = (255, 127, 80 )
    INDIAN_RED              = (205, 92 , 92 )
    LIGHT_CORAL             = (240, 128, 128)
    DARK_SALMON             = (233, 150, 122)
    SALMON                  = (250, 128, 114)
    LIGHT_SALMON            = (255, 160, 122)
    ORANGE_RED              = (255, 69 , 0  )
    DARK_ORANGE             = (255, 140, 0  )
    ORANGE                  = (255, 165, 0  )
    GOLD                    = (255, 215, 0  )
    DARK_GOLDEN_ROD         = (184, 134, 11 )
    GOLDEN_ROD              = (218, 165, 32 )
    PALE_GOLDEN_ROD         = (238, 232, 170)
    DARK_KHAKI              = (189, 183, 107)
    KHAKI                   = (240, 230, 140)
    OLIVE                   = (128, 128, 0  )
    YELLOW                  = (255, 255, 0  )
    YELLOW_GREEN            = (154, 205, 50 )
    DARK_OLIVE_GREEN        = (85 , 107, 47 )
    OLIVE_DRAB              = (107, 142, 35 )
    LAWN_GREEN              = (124, 252, 0  )
    CHART_REUSE             = (127, 255, 0  )
    GREEN_YELLOW            = (173, 255, 47 )
    DARK_GREEN              = (0  , 100, 0  )
    GREEN                   = (0  , 128, 0  )
    FOREST_GREEN            = (34 , 139, 34 )
    LIME                    = (0  , 255, 0  )
    LIME_GREEN              = (50 , 205, 50 )
    LIGHT_GREEN             = (144, 238, 144)
    PALE_GREEN              = (152, 251, 152)
    DARK_SEA_GREEN          = (143, 188, 143)
    MEDIUM_SPRING_GREEN     = (0  , 250, 154)
    SPRING_GREEN            = (0  , 255, 127)
    SEA_GREEN               = (46 , 139, 87 )
    MEDIUM_AQUA_MARINE      = (102, 205, 170)
    MEDIUM_SEA_GREEN        = (60 , 179, 113)
    LIGHT_SEA_GREEN         = (32 , 178, 170)
    DARK_SLATE_GRAY         = (47 , 79 , 79 )
    TEAL                    = (0  , 128, 128)
    DARK_CYAN               = (0  , 139, 139)
    AQUA                    = (0  , 255, 255)
    CYAN                    = (0  , 255, 255)
    LIGHT_CYAN              = (224, 255, 255)
    DARK_TURQUOISE          = (0  , 206, 209)
    TURQUOISE               = (64 , 224, 208)
    MEDIUM_TURQUOISE        = (72 , 209, 204)
    PALE_TURQUOISE          = (175, 238, 238)
    AQUA_MARINE             = (127, 255, 212)
    POWDER_BLUE             = (176, 224, 230)
    CADET_BLUE              = (95 , 158, 160)
    STEEL_BLUE              = (70 , 130, 180)
    CORN_FLOWER_BLUE        = (100, 149, 237)
    DEEP_SKY_BLUE           = (0  , 191, 255)
    DODGER_BLUE             = (30 , 144, 255)
    LIGHT_BLUE              = (173, 216, 230)
    SKY_BLUE                = (135, 206, 235)
    LIGHT_SKY_BLUE          = (135, 206, 250)
    MIDNIGHT_BLUE           = (25 , 25 , 112)
    NAVY                    = (0  , 0  , 128)
    DARK_BLUE               = (0  , 0  , 139)
    MEDIUM_BLUE             = (0  , 0  , 205)
    BLUE                    = (0  , 0  , 255)
    ROYAL_BLUE              = (65 , 105, 225)
    BLUE_VIOLET             = (138, 43 , 226)
    INDIGO                  = (75 , 0  , 130)
    DARK_SLATE_BLUE         = (72 , 61 , 139)
    SLATE_BLUE              = (106, 90 , 205)
    MEDIUM_SLATE_BLUE       = (123, 104, 238)
    MEDIUM_PURPLE           = (147, 112, 219)
    DARK_MAGENTA            = (139, 0  , 139)
    DARK_VIOLET             = (148, 0  , 211)
    DARK_ORCHID             = (153, 50 , 204)
    MEDIUM_ORCHID           = (186, 85 , 211)
    PURPLE                  = (128, 0  , 128)
    THISTLE                 = (216, 191, 216)
    PLUM                    = (221, 160, 221)
    VIOLET                  = (238, 130, 238)
    MAGENTA                 = (255, 0  , 255)
    ORCHID                  = (218, 112, 214)
    MEDIUM_VIOLET_RED       = (199, 21 , 133)
    PALE_VIOLET_RED         = (219, 112, 147)
    DEEP_PINK               = (255, 20 , 147)
    HOT_PINK                = (255, 105, 180)
    LIGHT_PINK              = (255, 182, 193)
    PINK                    = (255, 192, 203)
    ANTIQUE_WHITE           = (250, 235, 215)
    BEIGE                   = (245, 245, 220)
    BISQUE                  = (255, 228, 196)
    BLANCHED_ALMOND         = (255, 235, 205)
    WHEAT                   = (245, 222, 179)
    CORN_SILK               = (255, 248, 220)
    LEMON_CHIFFON           = (255, 250, 205)
    LIGHT_GOLDEN_ROD_YELLOW = (250, 250, 210)
    LIGHT_YELLOW            = (255, 255, 224)
    SADDLE_BROWN            = (139, 69 , 19 )
    SIENNA                  = (160, 82 , 45 )
    CHOCOLATE               = (210, 105, 30 )
    PERU                    = (205, 133, 63 )
    SANDY_BROWN             = (244, 164, 96 )
    BURLY_WOOD              = (222, 184, 135)
    TAN                     = (210, 180, 140)
    ROSY_BROWN              = (188, 143, 143)
    MOCCASIN                = (255, 228, 181)
    NAVAJO_WHITE            = (255, 222, 173)
    PEACH_PUFF              = (255, 218, 185)
    MISTY_ROSE              = (255, 228, 225)
    LAVENDER_BLUSH          = (255, 240, 245)
    LINEN                   = (250, 240, 230)
    OLD_LACE                = (253, 245, 230)
    PAPAYA_WHIP             = (255, 239, 213)
    SEA_SHELL               = (255, 245, 238)
    MINT_CREAM              = (245, 255, 250)
    SLATE_GRAY              = (112, 128, 144)
    LIGHT_SLATE_GRAY        = (119, 136, 153)
    LIGHT_STEEL_BLUE        = (176, 196, 222)
    LAVENDER                = (230, 230, 250)
    FLORAL_WHITE            = (255, 250, 240)
    ALICE_BLUE              = (240, 248, 255)
    GHOST_WHITE             = (248, 248, 255)
    HONEYDEW                = (240, 255, 240)
    IVORY                   = (255, 255, 240)
    AZURE                   = (240, 255, 255)
    SNOW                    = (255, 250, 250)
    BLACK                   = (0  , 0  , 0  )
    DIM_GRAY                = (105, 105, 105)
    GRAY                    = (128, 128, 128)
    DARK_GRAY               = (169, 169, 169)
    SILVER                  = (192, 192, 192)
    LIGHT_GRAY              = (211, 211, 211)
    GAINSBORO               = (220, 220, 220)
    WHITE_SMOKE             = (245, 245, 245)
    WHITE                   = (255, 255, 255)
    
    @staticmethod
    def values() -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in RGB]
    

class VideoFormat(Enum):
    AVI  = ".avi"
    M4V  = ".m4v"
    MKV  = ".mkv"
    MOV  = ".mov"
    MP4  = ".mp4"
    MPEG = ".mpeg"
    MPG  = ".mpg"
    WMV  = ".wmv"
    
    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "avi" : VideoFormat.AVI,
            "m4v" : VideoFormat.M4V,
            "mkv" : VideoFormat.MKV,
            "mov" : VideoFormat.MOV,
            "mp4" : VideoFormat.MP4,
            "mpeg": VideoFormat.MPEG,
            "mpg" : VideoFormat.MPG,
            "wmv" : VideoFormat.WMV,
        }

    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: VideoFormat.AVI,
            1: VideoFormat.M4V,
            2: VideoFormat.MKV,
            3: VideoFormat.MOV,
            4: VideoFormat.MP4,
            5: VideoFormat.MPEG,
            6: VideoFormat.MPG,
            7: VideoFormat.WMV,
        }
    
    @staticmethod
    def from_str(value: str) -> VideoFormat:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(VideoFormat.str_mapping, value.lower())
        return VideoFormat.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> VideoFormat:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(VideoFormat.int_mapping, value)
        return VideoFormat.int_mapping()[value]

    @staticmethod
    def from_value(value: Any) -> VideoFormat | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, VideoFormat):
            return value
        if isinstance(value, str):
            return VideoFormat.from_str(value)
        if isinstance(value, int):
            return VideoFormat.from_int(value)
        error_console.log(
            f"`value` must be `ImageFormat`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
    
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in VideoFormat]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in VideoFormat]
    

class VisionBackend(Enum):
    CV      = "cv"
    FFMPEG  = "ffmpeg"
    LIBVIPS = "libvips"
    PIL     = "pil"
    
    @staticmethod
    def str_mapping() -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "cv"     : VisionBackend.CV,
            "ffmpeg" : VisionBackend.FFMPEG,
            "libvips": VisionBackend.LIBVIPS,
            "pil"    : VisionBackend.PIL,
        }
    
    @staticmethod
    def int_mapping() -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: VisionBackend.CV,
            1: VisionBackend.FFMPEG,
            2: VisionBackend.LIBVIPS,
            3: VisionBackend.PIL,
        }
    
    @staticmethod
    def from_str(value: str) -> VisionBackend:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(VisionBackend.str_mapping, value.lower())
        return VisionBackend.str_mapping()[value]
    
    @staticmethod
    def from_int(value: int) -> VisionBackend:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(VisionBackend.int_mapping, value)
        return VisionBackend.int_mapping()[value]
    
    @staticmethod
    def from_value(value: Any) -> VisionBackend:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, VisionBackend):
            return value
        if isinstance(value, int):
            return VisionBackend.from_int(value)
        if isinstance(value, str):
            return VisionBackend.from_str(value)
        error_console.log(
            f"`value` must be `VisionBackend`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return VISION_BACKEND
    
    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in VisionBackend]
    
    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in VisionBackend]


# H2: - Globals ----------------------------------------------------------------

__current_file   = Path(__file__).absolute()        # "workspaces/one/src/one/core.py"
SOURCE_ROOT_DIR  = __current_file.parents[0]        # "workspaces/one/src/one"
CONTENT_ROOT_DIR = __current_file.parents[2]        # "workspaces/one"
PRETRAINED_DIR   = CONTENT_ROOT_DIR / "pretrained"  # "workspaces/one/pretrained"
DATA_DIR         = os.getenv("DATA_DIR", None)      # In case we have set value in os.environ
if DATA_DIR is None:
    DATA_DIR = Path("/data")                        # Run from Docker container
if not DATA_DIR.is_dir():
    DATA_DIR = CONTENT_ROOT_DIR / "data"            # Run from `one` package
if not DATA_DIR.is_dir():
    DATA_DIR = ""
    
    
DEFAULT_CROP_PCT = 0.875
IMG_MEAN         = [0.485, 0.456, 0.406]
IMG_STD          = [0.229, 0.224, 0.225]
PI               = torch.tensor(3.14159265358979323846)
VISION_BACKEND   = VisionBackend.PIL


# H1: - Conversion -------------------------------------------------------------

def _to_3d_array(input: np.ndarray) -> np.ndarray:
    """
    Convert an input to a 3D array.
    
    If the input is a 2D array, add a new axis at the beginning. If the input
    is a 4D array with the first dimension being 1, remove the first dimension.
    
    Args:
        input (np.ndarray): The input array.
    
    Returns:
        A 3D array of shape [H, W, C].
    """
    assert_numpy_of_ndim_in_range(input, 2, 4)
    if input.ndim == 2:    # [H, W] -> [1, H, W]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [H, W, C]
        pass
    elif input.ndim == 4 and input.shape[0] == 1:  # [1, H, W, C] -> [H, W, C]
        input = np.squeeze(input, axis=0)
    return input


def _to_3d_tensor(input: Tensor) -> Tensor:
    """
    Convert an input to a 3D tensor.
    
    If the input is a 2D tensor, add a new axis at the beginning. If the input
    is a 4D array with the first dimension being 1, remove the first dimension.
    
    Args:
        input (Tensor): The input tensor.
    
    Returns:
        A 3D tensor of shape [C, H, W].
    """
    assert_tensor_of_ndim_in_range(input, 2, 4)
    if input.ndim == 2:    # [H, W] -> [1, H, W]
        input = input.unsqueeze(dim=0)
    elif input.ndim == 3:  # [C, H, W]
        pass
    elif input.ndim == 4 and input.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
        input = input.squeeze(dim=0)
    return input


def _to_4d_array(input: np.ndarray) -> np.ndarray:
    """
    Convert an input to a 4D array.
    
    Args:
        input (np.ndarray): The input image.
    
    Returns:
        A 4D array of shape [B, H, W, C].
    """
    assert_numpy_of_ndim_in_range(input, 2, 5)
    if input.ndim == 2:    # [H, W] -> [1, 1, H, W]
        input = np.expand_dims(input, axis=0)
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [H, W, C] -> [1, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4:  # [B, H, W, C]
        pass
    elif input.ndim == 5 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)  # [1, B, H, W, C] -> [B, H, W, C]
    return input


def _to_4d_tensor(input: Tensor) -> Tensor:
    """
    Convert an input to a 4D tensor.
    
    Args:
        input (Tensor): The input tensor.
    
    Returns:
        A 4D tensor of shape [B, C, H, W].
    """
    assert_tensor_of_ndim_in_range(input, 2, 5)
    if input.ndim == 2:    # [H, W] -> [1, 1, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif input.ndim == 3:  # [C, H, W] -> [1, C, H, W]
        input = input.unsqueeze(dim=0)
    elif input.ndim == 4:  # [B, C, H, W]
        pass
    elif input.ndim == 5 and input.shape[0] == 1:
        input = input.squeeze(dim=0)  # [1, C, B, H, W] -> [B, C, H, W]
    return input


def _to_5d_array(input: np.ndarray) -> np.ndarray:
    """
    Convert an input to a 5D array.
    
    Args:
        input (np.ndarray): The input array.
    
    Returns:
        A 5D array of shape [*, B, C, H, W].
    """
    assert_numpy_of_ndim_in_range(input, 2, 6)
    if input.ndim == 2:    # [H, W] -> [1, 1, 1, H, W]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [H, W, C] -> [1, 1, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4:  # [B, H, W, C] -> [1, B, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 5:  # [*, B, H, W, C]
        pass
    elif input.ndim == 6 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0) # [1, *, B, H, W, C] -> [*, B, H, W, C]
    return input


def _to_5d_tensor(input: Tensor) -> Tensor:
    """
    Convert an input to a 5D tensor.
    
    Args:
        input (Tensor): The input tensor.
    
    Returns:
        A 5D tensor of shape [*, B, C, H, W].
    """
    assert_tensor_of_ndim_in_range(input, 2, 6)
    if input.ndim == 2:    # [H, W] -> [1, 1, 1, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif input.ndim == 3:  # [C, H, W] -> [1, 1, C, H, W]
        input = input.unsqueeze(dim=0)
        input = input.unsqueeze(dim=0)
    elif input.ndim == 4:  # [B, C, H, W] -> [1, B, C, H, W]
        input = input.unsqueeze(dim=0)
    elif input.ndim == 5:  # [*, B, C, H, W]
        pass
    elif input.ndim == 6 and input.shape[0] == 1:
        input = input.squeeze(dim=0)  # [1, *, B, C, H, W] -> [*, B, C, H, W]
    return input


def concat_lists(ll: list[list], inplace: bool = False) -> list:
    """
    Take a list of lists and returns a single list containing all the
    elements of the input list
    
    Args:
        ll (list[list]): List of lists.
        inplace (bool): If True, the original list is modified. If False, a
            copy is made. Defaults to False.
    
    Returns:
        A list of all the elements in the list of lists.
    """
    if not inplace:
        ll = ll.copy()
    return list(itertools.chain(*ll))


def copy_attr(a, b, include=(), exclude=()):
    """
    Copy all the attributes of object `b` to object `a`, except for those that
    start with an underscore, or are in the exclude list.
    
    Args:
        a: the object to copy to
        b: the object to copy from
        include: a tuple of attributes to include. If this is specified, only
            the attributes whose names are contained here will be copied over.
        exclude: A list of attributes to exclude from copying.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or \
            k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


@dispatch(int, Tensor)
def eye_like(n: int, input: Tensor) -> Tensor:
    """
    Create a tensor of shape `(n, n)` with ones on the diagonal and zeros
    everywhere else, and then repeats it along the batch dimension to match the
    shape of the input tensor.
    
    Args:
        n (int): The number of rows and columns in the output tensor.
        input (Tensor): The input tensor.
    
    Returns:
        A tensor of shape (input.shape[0], n, n).
    """
    if not n > 0:
        raise ValueError(f"Require `n` > 0. But got: {n}.")
    assert_tensor_of_atleast_ndim(input, 1)
    identity = torch.eye(n, device=input.device, dtype=input.dtype)
    return identity[None].repeat(input.shape[0], 1, 1)


def intersect_weight_dicts(da: dict, db: dict, exclude: tuple | list = ()) -> dict:
    """
    Take two dictionaries, and returns a new ordered dictionary that contains
    only the keys that are in both dictionaries, and whose values have the same
    shape.
    
    Args:
        da (dict): First dictionary.
        db (dict): Second dictionary.
        exclude (tuple | list): a list of strings that will be excluded from
            the intersection.
    
    Returns:
        A dictionary of the intersection of the two input dictionaries.
    """
    return {
        k: v for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def intersect_weight_ordered_dicts(
    da: OrderedDict, db: OrderedDict, exclude: tuple | list = ()
) -> OrderedDict:
    """
    Take two ordered dictionaries, and returns a new ordered dictionary that
    contains only the keys that are in both dictionaries, and whose values
    have the same shape.
    
    Args:
        da (OrderedDict): First ordered dictionary.
        db (OrderedDict): Second ordered dictionary.
        exclude (tuple | list): a list of strings that will be excluded from
            the intersection.
    
    Returns:
        An ordered dictionary of the intersection of the two input dictionaries.
    """
    return OrderedDict(
        (k, v) for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and v.shape == db[k].shape)
    )


def iter_to_iter(
    inputs     : Iterable,
    item_type  : type,
    return_type: type | None = None,
    inplace    : bool        = False,
):
    """
    Take an iterable, converts each item to a given type, and returns the
    result as an iterable of the same type as the input.
    
    Args:
        inputs (Iterable): The iterable to be converted.
        item_type (type): The type of the items in the iterable.
        return_type (type | None): Iterable type.
        inplace (bool): If True, the input iterable will be modified in place.
            Defaults to False.
    
    Returns:
        An iterable object of the inputs, cast to the item_type.
    """
    assert_iterable(inputs)
    assert_valid_type(item_type)
    
    if not inplace:
        inputs = copy(inputs)
        
    inputs = map(item_type, inputs)
    if return_type is None:
        return inputs
    else:
        return return_type(inputs)


def iter_to_list(inputs: Iterable, item_type: type, inplace: bool = False) -> list:
    """
    Take an iterable, and returns a list of the same items, but with the type
    of each item being the type specified by the `item_type` argument.
    
    Args:
        inputs (Iterable): The iterable to be converted.
        item_type (type): The type of the items in the iterable.
        inplace (bool): If True, the input will be modified in place.
            Defaults to False.
    
    Returns:
        A list of the inputs, cast to the item_type.
    """
    return iter_to_iter(
        inputs=inputs, item_type=item_type, return_type=list, inplace=inplace
    )


def iter_to_tuple(inputs: Iterable, item_type: type, inplace: bool = False):
    """
    Take an iterable, and returns a tuple of the same items, but with the type
    of each item being the type specified by the `item_type` argument.
    
    Args:
        inputs (Iterable): The iterable to be converted.
        item_type (type): The type of the items in the iterable.
        inplace (bool): If True, the input will be modified in place.
            Defaults to False.
    
    Returns:
        A tuple of the inputs, cast to the item_type.
    """
    return iter_to_iter(
        inputs=inputs, item_type=item_type, return_type=tuple, inplace=inplace
    )


def slice_list(input: list, lens: Ints) -> list[list]:
    """
    Takes a list and a list of integers, and returns a list of lists, where
    each sublist is of length specified by the corresponding integer in the
    list of integers.
    
    Args:
        input (list): list.
        lens (Ints): The number of elements in each sublist.
    
    Returns:
        A list of lists.
    """
    if isinstance(lens, int):
        assert_number_divisible_to(len(input), lens)
        lens = [lens] * int(len(input) / lens)
    
    assert_list(lens)
    assert_same_length(lens, input)
    
    out_list = []
    idx      = 0
    for i in range(len(lens)):
        out_list.append(input[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def to_3d_array(input) -> np.ndarray:
    """
    Convert input to a 3D array.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        A 3D array of shH, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 2 for i in input):
            input = np.stack(input)                                             # list[2D np.ndarray] -> 3D np.ndarray
        else:
            raise ValueError(f"Require `input.ndim` == 2.")
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray any dimensions
    if isinstance(input, np.ndarray):
        return _to_3d_array(input)                                              # np.ndarray any dimensions -> 3D np.ndarray
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_3d_array_list(input: Any) -> list[np.ndarray]:
    """
    Convert input to a list of 3D arrays.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        List of 3D arrays.
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # list[Tensor | np.ndarray]
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [input]                                                     # np.ndarray -> list[3D np.ndarray]
        elif input.ndim == 4:
            input = list(input)                                                 # np.ndarray -> list[3D np.ndarray]
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor any dimensions] -> list[np.ndarray any dimensions]
        
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        return _to_3d_array(input)                                              # list[np.ndarray any dimensions] -> list[3D np.ndarray]
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_3d_tensor(input: Any) -> Tensor:
    """
    Convert input to a 3D tensor.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        A 3D tensor of shape [C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 2 for i in input):
            input = torch.stack(input, dim=0)                                   # list[2D Tensor] -> 3D Tensor
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimensions
    if isinstance(input, Tensor):
        return _to_3d_tensor(input)                                             # Tensor any dimensions -> 3D Tensor
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_3d_tensor_list(input: Any) -> list[Tensor]:
    """
    Convert input to a list of 3D tensors.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        List of 3D tensors.
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimension
    if isinstance(input, Tensor):
        if input.ndim == 3:
            input = [input]                                                     # Tensor -> list[3D Tensor]
        elif input.ndim == 4:
            input = list(input)                                                 # Tensor -> list[3D Tensor]
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor any dimensions]
        
    if isinstance(input, list) and is_list_of(input, Tensor):
        return [_to_3d_tensor(i) for i in input]                                # list[Tensor any dimensions] -> list[3D Tensor]
    raise TypeError(f"Cannot convert `input` to a list of 3D tensor.")


def to_4d_array(input) -> np.ndarray:
    """
    Convert input to a 4D array.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        A 4D array of shape [B, H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):                                                
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and is_list_of(input, Tensor):                   
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 2 for i in input):                                     
            input = [np.expand_dims(i, axis=0) for i in input]                  # list[2D np.ndarray] -> list[3D np.ndarray]
        if all(i.ndim == 3 for i in input):                                     
            input = np.stack(input)                                             # list[3D np.ndarray] -> 4D np.ndarray
        else:                                                                   
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")                
    if isinstance(input, Tensor):                                               
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray any dimensions
    if isinstance(input, np.ndarray):                                           
        return _to_4d_array(input)                                              # np.ndarray any dimensions -> 4D np.ndarray
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    

def to_4d_array_list(input) -> list[np.ndarray]:
    """
    Convert input to a list of 4D arrays.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        List of 4D arrays of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # list[Tensor | np.ndarray]
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [np.expand_dims(input, axis=0)]                             # np.ndarray -> list[4D np.ndarray]
        elif input.ndim == 4:
            input = [input]                                                     # np.ndarray -> list[4D np.ndarray]
        elif input.ndim == 5:
            input = list(input)                                                 # np.ndarray -> list[4D np.ndarray]
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 5.")
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray any dimensions]
    
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        return [_to_4d_array(i) for i in input]                                 # list[np.ndarray any dimensions] -> list[4D np.ndarray]
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_4d_tensor(input: Any) -> Tensor:
    """
    Convert input to a 4D tensor.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        A 4D tensor of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]
    if isinstance(input, tuple):
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]                         # list[2D Tensor] -> list[3D Tensor]
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)                                   # list[3D Tensor] -> 4D Tensor
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimensions
    if isinstance(input, Tensor):
        return _to_4d_tensor(input)                                             # Tensor any dimensions -> 4D Tensor
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_4d_tensor_list(input) -> list[Tensor]:
    """
    Convert input to a list of 4D tensors.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        List of 4D tensors of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]              
    if isinstance(input, tuple):                                                                                                 
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]             
    if isinstance(input, np.ndarray):                                                                                            
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimension             
    if isinstance(input, Tensor):                                                                                                
        if input.ndim == 3:                                                                                                      
            input = [input.unsqueeze(dim=0)]                                    # Tensor -> list[4D Tensor]
        elif input.ndim == 4:                                                                                                    
            input = [input]                                                     # Tensor -> list[4D Tensor]
        elif input.ndim == 5:                                                                                                    
            input = list(input)                                                 # Tensor -> list[4D Tensor]
        else:                                                                                                                    
            raise ValueError(f"Require 3 <= `input.ndim` <= 5.")                                                                 
    if isinstance(input, list) and is_list_of(input, np.ndarray):                                                                
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor any dimensions]
    
    if isinstance(input, list) and is_list_of(input, Tensor):
        return [_to_4d_tensor(i) for i in input]                                # list[Tensor any dimensions] -> list[3D Tensor]
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_5d_array(input) -> np.ndarray:
    """
    Convert input to a 5D array.
    
    Args:
        input (Any): Input of arbitrary type.
    
    Returns:
        A 5D array of shape [*, B, H, W, C].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]         
    if isinstance(input, tuple):                                                                                            
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray]        
    if isinstance(input, list) and is_list_of(input, Tensor):                                                               
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor] -> list[np.ndarray]          
    if isinstance(input, list) and is_list_of(input, np.ndarray):                                                          
        if all(i.ndim == 2 for i in input):                                                                                 
            input = [np.expand_dims(i, axis=0) for i in input]                  # list[2D np.ndarray] -> list[3D np.ndarray]
        if all(3 <= i.ndim <= 4 for i in input):
            input = np.stack(input)                                             # list[3D np.ndarray] -> 4D np.ndarray      
        else:                                                                                                               
            raise ValueError(f"Require 2 <= `input.ndim` <= 4.")                                                            
    if isinstance(input, Tensor):                                                                                           
        input = input.detach().cpu().numpy()                                    # Tensor -> np.ndarray any dimensions       
    if isinstance(input, np.ndarray):                                                                                       
        return _to_5d_array(input)                                              # np.ndarray any dimensions -> 5D np.ndarray
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_5d_tensor(input: Any) -> Tensor:
    """
    Convert input to a 5D tensor.
    
    Args:
        input (Any): Input of arbitrary type.
    
    Returns:
        A 5D tensor of shape [*, B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())                                            # dict -> list[Tensor | np.ndarray]  
    if isinstance(input, tuple):                                                                                     
        input = list(input)                                                     # tuple -> list[Tensor | np.ndarray] 
    if isinstance(input, list) and is_list_of(input, np.ndarray):                                                    
        input = [torch.from_numpy(i) for i in input]                            # list[np.ndarray] -> list[Tensor]   
    if isinstance(input, list) and is_list_of(input, Tensor):                                                        
        if all(i.ndim == 2 for i in input):                                                                          
            input = [i.unsqueeze(dim=0) for i in input]                         # list[2D Tensor] -> list[3D Tensor] 
        if all(3 <= i.ndim <= 4 for i in input):                                                                     
            input = torch.stack(input, dim=0)                                   # list[3D Tensor] -> 4D Tensor       
        else:                                                                                                        
            raise ValueError(f"Require 2 <= `input.ndim` <= 4.")                                                     
    if isinstance(input, np.ndarray):                                                                                
        input = torch.from_numpy(input)                                         # np.ndarray -> Tensor any dimensions
    if isinstance(input, Tensor):                                                                                    
        return _to_5d_tensor(input)                                             # Tensor any dimensions -> 5D Tensor
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_list(input: Any) -> list:
    """
    Convert input into a list.
    
    Args:
        input (Any): Input of arbitrary type.
    
    Returns:
        A list of the input.
    """
    if isinstance(input, list):
        pass
    elif isinstance(input, tuple):
        input = list(input)
    elif isinstance(input, dict):
        input = [v for k, v in input.items()]
    else:
        input = [input]
    return input
    

def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """
    Take an integer n and returns a function that takes an iterable and returns
    a tuple of length n.
    
    Args:
        n (int): the number of elements in the tuple.
    
    Returns:
        A function that takes an integer and returns a tuple of that integer
        repeated n times.
    """
    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    
    return parse


def to_tuple(input: Any) -> tuple:
    """
    Convert input into a tuple.
    
    Args:
        input (Any): Input of arbitrary type.
    
    Returns:
        A tuple of the input.
    """
    if isinstance(input, list):
        input = tuple(input)
    elif isinstance(input, tuple):
        pass
    elif isinstance(input, dict):
        input = tuple([v for k, v in input.items()])
    else:
        input = tuple(input)
    return input


def to_size(size: Ints) -> tuple[int, int]:
    """
    Cast size object of any format into standard [H, W].
    
    Args:
        size (Ints): The size of the image to be generated.
    
    Returns:
        A tuple of size (H, W).
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:
            size = size[1:3]
        if len(size) == 1:
            size = (size[0], size[0])
    elif isinstance(size, int):
        size = (size, size)
    return tuple(size)


@dispatch(np.ndarray)
def upcast(input: np.ndarray) -> np.ndarray:
    """
    Protects from numerical overflows in multiplications by upcasting to
    the equivalent higher type.
    
    Args:
        input (Tensor): Array of arbitrary type.
    
    Returns:
        Array of higher type.
    """
    if not isinstance(input, np.ndarray):
        raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")
    if type(input) in (np.float16, np.float32, np.float64):
        return input.astype(float)
    if type(input) in (np.int16, np.int32, np.int64):
        return input.astype(int)
    return input


@dispatch(Tensor)
def upcast(input: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to
    the equivalent higher type.
    
    Args:
        input (Tensor): Tensor of arbitrary type.
    
    Returns:
        Tensor of higher type.
    """
    assert_tensor(input)
    if input.dtype in (torch.float16, torch.float32, torch.float64):
        return input.to(torch.float)
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return input.to(torch.int)
    return input


@dispatch(int, Tensor)
def vec_like(n: int, input: Tensor) -> Tensor:
    """
    Create a vector of zeros with the same shape as the input.
    
    Args:
        n (int): The number of elements in the vector.
        input (Tensor): The input tensor.
    
    Returns:
        A tensor of zeros with the same shape as the input tensor.
    """
    if not n > 0:
        raise ValueError(f"Require `n` > 0. But got: {n}.")
    assert_tensor_of_atleast_ndim(input, 1)
    vec = torch.zeros(n, 1, device=input.device, dtype=input.dtype)
    return vec[None].repeat(input.shape[0], 1, 1)


@dispatch(list)
def unique(input: list) -> list:
    """
    Return a list with only unique items.

    Args:
        input (list): list.
    
    Returns:
        A list containing only unique items.
    """
    return list(set(input))


@dispatch(tuple)
def unique(input: tuple) -> tuple:
    """
    Return a tuple containing the unique elements of the given tuple.
    
    Args:
        input (tuple): tuple.
    
    Returns:
        A tuple of unique elements from the input tuple.
    """
    return tuple(set(input))


to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)
to_5tuple = to_ntuple(5)
to_6tuple = to_ntuple(6)


# H1: - Dataclass --------------------------------------------------------------

def majority_voting(labels: list[dict]) -> dict:
    """
    It counts the number of appearance of each label, and returns the label with
    the highest count.
    
    Args:
        labels (list[dict]): List of object's label.
    
    Returns:
        A dictionary of the label that has the most votes.
    """
    # Count number of appearance of each label.
    unique_labels = Munch()
    label_voting  = Munch()
    for label in labels:
        k = label.get("id")
        v = label_voting.get(k)
        if v:
            label_voting[k] = v + 1
        else:
            unique_labels[k] = label
            label_voting[k]  = 1
    
    # Get k (label's id) with max v
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]


class BBox:
    """
    Bounding box object with (b1, b2, b3, b4, confidence) format.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
    """
    
    def __init__(
        self,
        b1        : float,
        b2        : float,
        b3        : float,
        b4        : float,
        confidence: float            = 1.0,
        id        : int | str        = uuid.uuid4().int,
        image_id  : int | str | None = None,
        class_id  : int | str | None = None,
        format    : BBoxFormat       = BBoxFormat.CXCYWH_NORM,
    ):
        self.id         = id
        self.image_id   = image_id
        self.class_id   = class_id
        self.b1         = b1
        self.b2         = b2
        self.b3         = b3
        self.b4         = b4
        self.confidence = confidence
        self.format     = format
        
    @property
    def is_normalized(self) -> bool:
        """
        It checks if the values of the four variables are less than or equal
        to 1.0.
        
        Returns:
          A boolean value.
        """
        return all(i <= 1.0 for i in [self.b1, self.b2, self.b3, self.b4])
    
    @property
    def label(self) -> Tensor:
        """
        It returns a tensor containing the image id, class id, bounding box
        coordinates, and confidence
        
        Returns:
            A tensor of the image_id, class_id, b1, b2, b3, b4, and confidence.
        """
        return torch.Tensor(
            [
                self.image_id,
                self.class_id,
                self.b1, self.b2, self.b3, self.b4,
                self.confidence
            ],
            dtype=torch.float32
        )


class ClassLabel:
    """
    ClassLabel is a list of all classes' dictionaries in the dataset.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/ClassLabel

    Attributes:
        classes (list[dict]):
            List of all classes in the dataset.
    """

    def __init__(self, classes: list[dict]):
        assert_list_of(classes, item_type=dict)
        self._classes = classes

    @staticmethod
    def from_dict(d: dict) -> ClassLabel:
        """
        It takes a dictionary and returns a ClassLabel object.
        
        Args:
            d (dict): dict.
        
        Returns:
            A ClassLabel object.
        """
        assert_dict_contain_key(d, "classes")
        classes = d["classes"]
        classes = Munch.fromDict(classes)
        return ClassLabel(classes=classes)
        
    @staticmethod
    def from_file(path: Path_) -> ClassLabel:
        """
        It creates a ClassLabel object from a `json` file.
        
        Args:
            path (Path_): The path to the `json` file.
        
        Returns:
            A ClassLabel object.
        """
        assert_json_file(path)
        return ClassLabel.from_dict(load_from_file(path))
    
    @staticmethod
    def from_value(value: Any) -> ClassLabel | None:
        """
        It converts an arbitrary value to a ClassLabel.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            A ClassLabel object.
        """
        if isinstance(value, ClassLabel):
            return value
        if isinstance(value, (dict, Munch)):
            return ClassLabel.from_dict(value)
        if isinstance(value, (str, Path)):
            return ClassLabel.from_file(value)
        error_console.log(
            f"`value` must be `ClassLabel`, `dict`, `str`, or `Path`. "
            f"But got: {type(value)}."
        )
        return None
        
    @property
    def classes(self) -> list:
        return self._classes

    def color_legend(self, height: int | None = None) -> Tensor:
        """
        It creates a legend of the classes in the dataset.
        
        Args:
            height (int | None): The height of the legend. If None, the legend
                will be 25px high per class.
        
        Returns:
            A tensor of the legend.
        """
        from one.vision.acquisition import to_tensor
        
        num_classes = len(self.classes)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(((num_classes * row_height) + 25, 300, 3), dtype=np.uint8)

        # Loop over the class names + colors
        for i, label in enumerate(self.classes):
            color = label.color  # Draw the class name + color on the legend
            color = color[::-1]  # Convert to BGR format since OpenCV operates on BGR format.
            cv2.putText(
                img       = legend,
                text      = label.name,
                org       = (5, (i * row_height) + 17),
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color     = (0, 0, 255),
                thickness = 2
            )
            cv2.rectangle(
                img       = legend,
                pt1       = (150, (i * 25)),
                pt2       = (300, (i * row_height) + 25),
                color     = color,
                thickness = -1
            )
        return to_tensor(image=legend)
        
    def colors(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """
        Returns a list of colors for each class in the dataset.
        
        Args:
            key (str): The key to search for. Defaults to id.
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            A list of colors.
        """
        labels_colors = []
        for label in self.classes:
            if hasattr(label, key) and hasattr(label, "color"):
                if (exclude_negative_key and label[key] <  0  ) or \
                   (exclude_max_key      and label[key] >= 255):
                    continue
                labels_colors.append(label.color)
        return labels_colors

    @property
    def id2label(self) -> dict[int, dict]:
        """
        
        Returns:
            A dictionary with the label id as the key and the label as the
            value.
        """
        return {label["id"]: label for label in self.classes}

    def ids(
        self,
        key                 : str = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """
        Returns a list of all the ids of the classes in the class list.
        
        Args:
            key (str): The key to search for. Defaults to id.
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            A list of ids.
        """
        ids = []
        for c in self.classes:
            if hasattr(c, key):
                if (exclude_negative_key and c[key] <  0  ) or \
                   (exclude_max_key      and c[key] >= 255):
                    continue
                ids.append(c[key])
        return ids

    @property
    def list(self) -> list:
        return self.classes

    @property
    def name2label(self) -> dict[str, dict]:
        """
        
        Returns:
            A dictionary with the label name as the key and the label as the
            value.
        """
        return {c["name"]: c for c in self.classes}

    def names(
        self,
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """
        It returns a list of names of the classes in the dataset.
        
        Args:
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            A list of names of the classes.
        """
        names = []
        for c in self.classes:
            if hasattr(c, "id"):
                if (exclude_negative_key and c["id"] <  0  ) or \
                   (exclude_max_key      and c["id"] >= 255):
                    continue
                names.append(c["name"])
            else:
                names.append("")
        return names
    
    def num_classes(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> int:
        """
        Count the number of classes in the dataset, excluding the negative and
        max classes if specified.
        
        Args:
            key (str): The key to search for. Defaults to id.
            exclude_negative_key (bool): If True, the negative value of the key
                will be excluded. Defaults to True.
            exclude_max_key (bool): If True, the maximum value of the key will
                be excluded. Defaults to True.
        
        Returns:
            The number of classes in the dataset.
        """
        count = 0
        for c in self.classes:
            if hasattr(c, key):
                if (exclude_negative_key and c[key] <  0  ) or \
                   (exclude_max_key      and c[key] >= 255):
                    continue
                count += 1
        return count

    def get_class(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> dict | None:
        """
        Returns the class with the given key and value, or None if no such
        class exists.
        
        Args:
            key (str): The key to search for. Defaults to id.
            value (int | str | None): The value of the key to search for.
                Defaults to None.
        
        Returns:
            A dictionary of the class that matches the key and value.
        """
        for c in self.classes:
            if hasattr(c, key) and (value == c[key]):
                return c
        return None
    
    def get_class_by_name(self, name: str) -> dict | None:
        """
        Returns the class with the given class name, or None if no such class
        exists.
        
        Args:
            name (str): The name of the class you want to get.
        
        Returns:
            A dictionary of the class with the given name.
        """
        return self.get_class(key="name", value=name)
    
    def get_id(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> int | None:
        """
        Returns the id of the class label that matches the given key and value.
        
        Args:
           key (str): The key to search for. Defaults to id.
           value (int | str | None): The value of the key to search for.
                Defaults to None.
        
        Returns:
            The id of the class.
        """
        class_label: dict = self.get_class(key=key, value=value)
        return class_label["id"] if class_label is not None else None
    
    def get_id_by_name(self, name: str) -> int | None:
        """
        Given a class name, return the class id.
        
        Args:
            name (str): The name of the class you want to get the ID of.
        
        Returns:
            The id of the class.
        """
        class_label = self.get_class_by_name(name=name)
        return class_label["id"] if class_label is not None else None
    
    def get_name(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> str | None:
        """
        Get the name of a class given a key and value.
        
        Args:
            key (str): The key to search for. Defaults to id.
            value (int | str | None): The value of the key to search for.
                Defaults to None.
        
        Returns:
            The name of the class.
        """
        c = self.get_class(key=key, value=value)
        return c["name"] if c is not None else None
       
    def show_color_legend(self, height: int | None = None):
        """Show a pretty color lookup legend using OpenCV drawing functions.

        Args:
            height (int | None): Height of the color legend image.
                Defaults to None.
        """
        color_legend = self.color_legend(height=height)
        plt.imshow(color_legend.permute(1, 2, 0))
        plt.title("Color Legend")
        plt.show()
        
    def print(self):
        """
        Print all classes using `rich` package.
        """
        if not (self.classes and len(self.classes) > 0):
            console.log("[yellow]No class is available.")
            return
        
        console.log("[red]Classlabel:")
        print_table(self.classes)


class Image:
    """
    Image object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image
    
    Args:
        id (int | str): The id of the image. This can be an integer or a string.
            This attribute is useful for batch processing where you want to keep
            the objects in the correct frame sequence.
        name (str | None): The name of the image. Defaults to None.
        path (Path_ | None): The path to the image file. Defaults to None.
        image (Tensor[*, C, H, W] | None): The image to be loaded.
            Defaults to None.
        load_on_create (bool): If True, the image will be loaded into memory
            when the object is created. Defaults to False.
        keep_in_memory (bool): If True, the image will be loaded into memory
            and kept there. Defaults to False.
        backend (VisionBackend_): The backend to use for image processing.
            Defaults to VISION_BACKEND.
    """
    
    def __init__(
        self,
        id            : int | str      = uuid.uuid4().int,
        name          : str | None     = None,
        path          : Path_ | None   = None,
        image         : Tensor | None  = None,
        load_on_create: bool           = False,
        keep_in_memory: bool           = False,
        backend       : VisionBackend_ = VISION_BACKEND,
    ):
        from one.vision.acquisition import get_image_shape
        
        self.id             = id
        self.image          = None
        self.keep_in_memory = keep_in_memory
        self.backend        = backend
        
        if path is not None:
            path = Path(path)
            assert_image_file(path)
        self.path = path
        
        if name is None:
            name = str(Path(path).name) if is_image_file(path=path) else f"{id}"
        self.name = name
        
        if load_on_create and image is None:
            image = self.load()

        self.shape = get_image_shape(image) if image is not None else None

        if self.keep_in_memory:
            self.image = image
    
    def load(
        self, path: Path_ | None = None, keep_in_memory: bool = False,
    ) -> Tensor:
        """Load image into memory.
        
        Args:
            path (Path_ | None):
                The path to the image file. Defaults to None.
            keep_in_memory (bool):
                If True, the image will be loaded into memory and kept there.
                Defaults to False.
            
        Returns:
            Return image Tensor of shape [1, C, H, W] to caller.
        """
        from one.vision.acquisition import read_image
        from one.vision.acquisition import get_image_shape
    
        self.keep_in_memory = keep_in_memory
        
        if is_image_file(path):
            self.path = Path(path)
        assert_image_file(path=self.path)
        
        image      = read_image(path=self.path, backend=self.backend)
        self.shape = get_image_shape(image=image) if (image is not None) else self.shape
        
        if self.keep_in_memory:
            self.image = image
        
        return image
        
    @property
    def meta(self) -> dict:
        """
        It returns a dictionary of metadata about the object.
        
        Returns:
            A dictionary with the id, name, path, and shape of the image.
        """
        return {
            "id"   : self.id,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }


class KITTILabel:
    """
    
    """
    pass


class VOCBBox(BBox):
    """
    VOC bounding box object.
    
    References:
        https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBox
    
    Args:
        name (int | str): This is the name of the object that we are trying to
            identify (i.e., class_id).
        truncated (int): Indicates that the bounding box specified for the
            object does not correspond to the full extent of the object.
            For example, if an object is visible partially in the image then
            we set truncated to 1. If the object is fully visible then set
            truncated to 0.
        difficult (int): An object is marked as difficult when the object is
            considered difficult to recognize. If the object is difficult to
            recognize then we set difficult to 1 else set it to 0.
        bndbox (Tensor | list | tuple): Axis-aligned rectangle specifying the
            extent of the object visible in the image.
        pose (str): Specify the skewness or orientation of the image.
            Defaults to Unspecified, which means that the image is not skewed.
    """
    
    def __init__(
        self,
        name     : str,
        truncated: int,
        difficult: int,
        bndbox   : Tensor | list | tuple,
        pose     : str = "Unspecified",
        *args, **kwargs
    ):
        if isinstance(bndbox, Tensor):
            assert_tensor_of_ndim(bndbox, 1)
            bndbox = bndbox.tolist()
        if isinstance(bndbox, (list, tuple)):
            assert_sequence_of_length(bndbox, 4)
        super().__init__(
            b1 = bndbox[0],
            b2 = bndbox[1],
            b3 = bndbox[2],
            b4 = bndbox[3],
            *args, **kwargs
        )
        self.name      = name
        self.pose      = pose
        self.truncated = truncated
        self.difficult = difficult
        
    def convert_name_to_id(self, class_labels: ClassLabel):
        """
        If the class name is a number, then it is the class id.
        Otherwise, the class id is searched from the ClassLabel object.
        
        Args:
            class_labels (ClassLabel): The ClassLabel containing all classes
                in the dataset.
        """
        self.class_id = int(self.name) \
            if self.name.isnumeric() \
            else class_labels.get_id(key="name", value=self.name)


class VOCLabel:
    """
    VOC label.
    
    Args:
        folder (str): Folder that contains the images.
        filename (str): Name of the physical file that exists in the folder.
        path (Path_): The absolute path where the image file is present.
        source (dict): Specifies the original location of the file in a
            database. Since we do not use a database, it is set to `Unknown`
            by default.
        size (dict): Specify the width, height, depth of an image. If the image
            is black and white then the depth will be 1. For color images,
            depth will be 3.
        segmented (int): Signify if the images contain annotations that are
            non-linear (irregular) in shape - commonly referred to as polygons.
            Defaults to 0 (linear shape).
        object (dict | list | None): Contains the object details. If you have
            multiple annotations then the object tag with its contents is
            repeated. The components of the object tags are:
            - name (int, str): This is the name of the object that we are
                trying to identify (i.e., class_id).
            - pose (str): Specify the skewness or orientation of the image.
                Defaults to `Unspecified`, which means that the image is not
                skewed.
            - truncated (int): Indicates that the bounding box specified for
                the object does not correspond to the full extent of the object.
                For example, if an object is visible partially in the image
                then we set truncated to 1. If the object is fully visible then
                set truncated to 0.
            - difficult (int): An object is marked as difficult when the object
                is considered difficult to recognize. If the object is
                difficult to recognize then we set difficult to 1 else set it
                to 0.
            - bndbox (dict): Axis-aligned rectangle specifying the extent of
                the object visible in the image.
        class_labels (ClassLabel | None): ClassLabel object. Defaults to None.
    """
    
    def __init__(
        self,
        folder      : str,
        filename    : str,
        path        : Path_,
        source      : dict,
        size        : dict,
        segmented   : int,
        object      : dict | list | None,
        class_labels: ClassLabel | None = None,
        *args, **kwargs
    ):
        from one.vision.shape import box_xyxy_to_cxcywh_norm
        
        self.folder    = folder
        self.filename  = filename
        self.path      = Path(path)
        self.source    = source
        self.size      = size
        self.width     = int(self.size.get("width",  0))
        self.height    = int(self.size.get("height", 0))
        self.depth     = int(self.size.get("depth",  0))
        self.segmented = segmented

        if object is None:
            object = []
        else:
            object = [object] if not isinstance(object, dict) else object
        assert_list_of(object, dict)
        
        for i, o in enumerate(object):
            bndbox   = o.get("bndbox")
            box_xyxy = torch.FloatTensor([
                int(bndbox["xmin"]), int(bndbox["ymin"]),
                int(bndbox["xmax"]), int(bndbox["ymax"])
            ])
            o["bndbox"] = box_xyxy_to_cxcywh_norm(box_xyxy, self.height, self.width)
            o["format"] = BBoxFormat.CXCYWH_NORM
        self.bboxes = [VOCBBox(*b) for b in object]
        
        if isinstance(class_labels, ClassLabel):
            self.convert_names_to_ids(class_labels=class_labels)
    
    @staticmethod
    def create_from_dict(d: dict, *args, **kwargs) -> VOCLabel:
        """
        Create a VOCLabel object from a dictionary.
        
        Args:
            d (dict):
                Dictionary containing VOC data.
        
        Returns:
            A VOCLabel object.
        """
        assert_dict_contain_key(d, "annotation")
        d = d["annotation"]
        return VOCLabel(
            folder    = d.get("folder"   , ""),
            filename  = d.get("filename" , ""),
            path      = d.get("path"     , ""),
            source    = d.get("source"   , {"database": "Unknown"}),
            size      = d.get("size"     , {"width": 0, "height": 0, "depth": 3}),
            segmented = d.get("segmented", 0),
            object    = d.get("object"   , []),
            *args, **kwargs
        )
        
    @staticmethod
    def create_from_file(path: Path_, *args, **kwargs) -> VOCLabel:
        """
        Load VOC label from file.
        
        Args:
            path (Path): Annotation file.
            
        Returns:
            A VOCLabel object.
        """
        assert_xml_file(path)
        return VOCLabel.create_from_dict(load_from_file(path), *args, **kwargs)
   
    def convert_names_to_ids(
        self, class_labels: ClassLabel, parallel: bool = False
    ):
        """
        Convert `name` property in each `object` to class id.
        
        Args:
            class_labels (ClassLabel): The ClassLabel containing all classes
                in the dataset.
            parallel (bool): If True, run parallely. Defaults to False.
        """
        if parallel:
            def run(i):
                self.bboxes[i].convert_name_to_id(class_labels)
            
            Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
                delayed(run)(i) for i in range(len(self.bboxes))
            )
        else:
            for o in self.bboxes:
                o.convert_name_to_id(class_labels=class_labels)


# H1: - Dataset ----------------------------------------------------------------

class Dataset(data.Dataset, metaclass=ABCMeta):
    """
    Base class for making datasets. It is necessary to override the
    `__getitem__` and `__len__` method.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.root    = Path(root)
        self.split   = split
        self.shape   = shape
        self.verbose = verbose
        
        if transform is not None:
            transform = ComposeTransform(transform)
        if target_transform is not None:
            target_transform = ComposeTransform(target_transform)
        if transforms is not None:
            transforms = ComposeTransform(transforms)
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        """
        has_transforms         = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can be passed "
                "as argument."
            )

        self.transform        = transform
        self.target_transform = target_transform
        
        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        """
        
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index (int): The index of the sample to be retrieved.
        
        Returns:
            Any.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        
        Returns:
            Length of the dataset.
        """
        pass
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        """
        This function is used to print a string representation of the object.
        """
        return ""


class DataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """
    Base class for all datamodules.
    
    Args:
        root (Path_): Root directory of dataset.
        name (str): Dataset's name.
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        batch_size (int): Number of samples in one forward & backward pass.
            Defaults to 1.
        devices (Device): The devices to use. Defaults to 0.
        shuffle (bool): If True, reshuffle the data at every training epoch.
             Defaults to True.
        collate_fn (Callable | None): Collate function used to fused input items
            together when using `batch_size > 1`.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        name            : str,
        shape           : Ints,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        batch_size      : int                = 1,
        devices         : Devices            = 0,
        shuffle         : bool               = True,
        collate_fn      : Callable | None    = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__()
        self.root             = Path(root)
        self.name             = name
        self.shape            = shape
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        self.batch_size       = batch_size
        self.devices          = devices
        self.shuffle          = shuffle
        self.collate_fn       = collate_fn
        self.verbose          = verbose
        self.dataset_kwargs   = kwargs
        self.train            = None
        self.val              = None
        self.test             = None
        self.predict          = None
        self.class_label      = None
       
    @property
    def devices(self) -> list:
        """
        Returns a list of devices.
        """
        return self._devices

    @devices.setter
    def devices(self, devices: Devices):
        self._devices = to_list(devices)
    
    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        if isinstance(self.class_label, ClassLabel):
            return self.class_label.num_classes()
        return 0
    
    @property
    def num_workers(self) -> int:
        """
        Returns number of workers used in the data loading pipeline.
        """
        # Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 4 * len(self.devices)
        # return 4  # os.cpu_count()

    @property
    def train_dataloader(self) -> TrainDataLoaders | None:
        """
        If the train set exists, return a DataLoader object with the train set,
        otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.train:
            return DataLoader(
                dataset            = self.train,
                batch_size         = self.batch_size,
                shuffle            = self.shuffle,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def val_dataloader(self) -> EvalDataLoaders | None:
        """
        If the validation set exists, return a DataLoader object with the
        validation set, otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.val:
            return DataLoader(
                dataset            = self.val,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def test_dataloader(self) -> EvalDataLoaders | None:
        """
        If the test set exists, return a DataLoader object with the  test set,
        otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.test:
            return DataLoader(
                dataset            = self.test,
                batch_size         = 1,  # self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def predict_dataloader(self) -> Union[EvalDataLoaders, None]:
        """
        If the prediction set exists, return a DataLoader object with the
        prediction set, otherwise return None
        
        Returns:
            A DataLoader object.
        """
        if self.predict:
            return DataLoader(
                dataset            = self.predict,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None
    
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        pass
    
    @abstractmethod
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        pass

    @abstractmethod
    def load_class_label(self):
        """
        Load ClassLabel.
        """
        pass
        
    def summarize(self):
        """
        It prints a summary table of the datamodule.
        """
        table = Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Desc")
        
        table.add_row("1", "train",        f"{len(self.train)              if self.train is not None else None}")
        table.add_row("2", "val",          f"{len(self.val)                if self.val   is not None else None}")
        table.add_row("3", "test",         f"{len(self.test)               if self.test  is not None else None}")
        table.add_row("4", "class_labels", f"{self.class_label.num_classes if self.class_label is not None else None}")
        table.add_row("5", "batch_size",   f"{self.batch_size}")
        table.add_row("6", "shape",        f"{self.shape}")
        table.add_row("7", "num_workers",  f"{self.num_workers}")
        console.log(table)


# H2: - Unlabeled --------------------------------------------------------------

class UnlabeledDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of data
    samples.
    """
    pass


class UnlabeledImageDataset(UnlabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of images.
    This is mainly used for unsupervised learning tasks.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root        : Path_,
        split       : str,
        shape       : Ints,
        transform   : Transforms_ | None = None,
        transforms  : Transforms_ | None = None,
        cache_data  : bool               = False,
        cache_images: bool               = False,
        backend     : VisionBackend_     = VISION_BACKEND,
        verbose     : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root       = root,
            split      = split,
            shape      = shape,
            transform  = transform,
            transforms = transforms,
            verbose    = verbose,
            *args, **kwargs
        )
        self.backend = VisionBackend.from_value(backend)
        
        self.images: list[Image] = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.list_images()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
        
        self.filter()
        self.verify()
        if cache_data or not cache_file.is_file():
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
        
    def __getitem__(self, index: int) -> tuple[Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.

		Args:
			index (int): The index of the sample to be retrieved.

		Returns:
			input (Tensor[1, C, H, W]): Sample, optionally transformed by the
			    respective transforms.
			meta (dict): Metadata of image.
		"""
        item  = self.images[index]
        input = item.image if item.image is not None else item.load()
        meta  = item.meta
        
        if self.transform is not None:
            input, *_ = self.transform(input=input, target=None, dataset=self)
        if self.transforms is not None:
            input, *_ = self.transforms(input=input, target=None, dataset=self)
        return input, meta
        
    def __len__(self) -> int:
        """
        This function returns the length of the images list.
        
        Returns:
            The length of the images list.
        """
        return len(self.images)
        
    @abstractmethod
    def list_images(self):
        """
        List image files.
        """
        pass
    
    @abstractmethod
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass
    
    def verify(self):
        """
        Verify and checking.
        """
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        console.log(f"Number of samples: {len(self.images)}.")
    
    def cache_data(self, path: Path_):
        """
        Cache data to `path`.
        
        Args:
            path (Path_): The path to save the cache.
        """
        cache = {"images": self.images}
        torch.save(cache, str(path))
    
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, meta).
        """
        input, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input, 0)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4.")
        return input, meta


class UnlabeledVideoDataset(UnlabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of video.
    This is mainly used for unsupervised learning tasks.
    """
    pass


class ImageDirectoryDataset(UnlabeledImageDataset):
    """
    A directory of images starting from `root` directory.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root        : Path_,
        split       : str,
        shape       : Ints,
        transform   : Transforms_ | None = None,
        transforms  : Transforms_ | None = None,
        cache_data  : bool               = False,
        cache_images: bool               = False,
        backend     : VisionBackend_     = VISION_BACKEND,
        verbose     : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            shape        = shape,
            transform    = transform,
            transforms   = transforms,
            cache_data   = cache_data,
            cache_images = cache_images,
            backend      = backend,
            verbose      = verbose,
            *args, **kwargs
        )
        
    def list_images(self):
        """
        List image files.
        """
        assert_dir(self.root)
        
        with progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                pattern.rglob("*"),
                description=f"[bright_yellow]Listing {self.split} images"
            ):
                if is_image_file(path):
                    self.images.append(Image(path=path, backend=self.backend))
                    
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass


# H2: - Labeled ----------------------------------------------------------------

class LabeledDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of data
    samples.
    """
    pass


class LabeledImageDataset(LabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of images.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            verbose          = verbose,
            *args, **kwargs
        )
        self.backend     = VisionBackend.from_value(backend)
        self.class_label = ClassLabel.from_value(class_label)
        
        if not hasattr(self, "images"):
            self.images: list[Image] = []
        if not hasattr(self, "labels"):
            self.labels = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.list_images()
            self.list_labels()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
            self.labels = cache["labels"]
            
        self.filter()
        self.verify()
        if cache_data or not cache_file.is_file():
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
    
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Any, Image]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
		Args:
          index (int): The index of the sample to be retrieved.

		Returns:
			input (Tensor[1, C, H, W]): Input sample, optionally transformed by
			    the respective transforms.
			target (Any): Target, depended on label type.
			meta (Image): Meta data of image.
		"""
        pass
    
    def __len__(self) -> int:
        """
        This function returns the length of the images list.
        
        Returns:
            The length of the images list.
        """
        return len(self.images)
    
    @abstractmethod
    def list_images(self):
        """
        List image files.
        """
        pass

    @abstractmethod
    def list_labels(self):
        """
        List label files.
        """
        pass

    @abstractmethod
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass

    def verify(self):
        """
        Verify and checking.
        """
        if not (is_same_length(self.images, self.labels) and len(self.images) > 0):
            raise RuntimeError(
                f"Number of `images` and `labels` must be the same. "
                f"But got: {len(self.images)} != {len(self.labels)}"
            )
        console.log(f"Number of {self.split} samples: {len(self.images)}.")
        
    def cache_data(self, path: Path_):
        """
        Cache data to `path`.
        
        Args:
            path (Path_): The path to save the cache.
        """
        cache = {
            "images": self.images,
            "labels": self.labels,
        }
        torch.save(cache, str(path))
    
    @abstractmethod
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        pass


class LabeledVideoDataset(LabeledDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent an unlabeled collection of video.
    """
    pass


# H2: - Classification ---------------------------------------------------------

class ImageClassificationDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated classification
    labels stored in a simple JSON format.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[int] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def __getitem__(self, index: int) -> tuple[Tensor, int, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
		Args:
          index (int): The index of the sample to be retrieved.

		Returns:
			input (Tensor[1, C, H, W]): Input sample, optionally transformed
			    by the respective transforms.
			target (int): Classification labels.
			meta (dict): Metadata of image.
		"""
        item   = self.images[index]
        input  = item.image if item.image is not None else item.load()
        target = self.labels[index]
        meta   = item.meta
        
        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")


class VideoClassificationDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """
    Base type for datasets that represent a collection of videos and a set of
    associated classification labels.
    """
    pass


class ImageClassificationDirectoryTree(ImageClassificationDataset):
    """
    A directory tree whose sub-folders define an image classification dataset.
    """
    
    def list_images(self):
        """
        List image files.
        """
        pass

    def list_labels(self):
        """
        List label files.
        """
        pass
    
    def filter(self):
        """
        Filter unwanted samples.
        """
        pass


# H2: - Detection --------------------------------------------------------------

class ImageDetectionDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of images and a set
    of associated detections.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[list[BBox]] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
		Args:
          index (int): The index of the sample to be retrieved.

		Returns:
			input (Tensor[1, C, H, W]): Input sample, optionally transformed by
			    the respective transforms.
			target (Tensor[N, 7]): Bounding boxes.
			meta (Image): Metadata of image.
		"""
        item   = self.images[index]
        input  = item.image if item.image is not None else item.load()
        target = self.labels[index]
        target = torch.stack([b.label for b in target])
        meta   = item.meta

        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input  = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input  = torch.cat(input,  0)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        for i, l in enumerate(target):
            l[:, 0] = i  # add target image index for build_targets()
        return input, target, meta

    
class VideoDetectionDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """
    Base type for datasets that represent a collection of videos and a set of 
    associated video detections.
    """
    pass


class COCODetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated object 
    detections saved in `COCO Object Detection Format 
    <https://cocodataset.org/#format-data>`.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """
        List label files.
        """
        json = self.annotation_file()
        assert_json_file(json)
        json_data = load_from_file(json)
    
    @abstractmethod
    def annotation_file(self) -> Path_:
        """
        Returns the path to json annotation file.
        """
        pass


class VOCDetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """
    A labeled dataset consisting of images and their associated object
    detections saved in `PASCAL VOC format
    <http://host.robots.ox.ac.uk/pascal/VOC>`.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_images    : bool               = False,
        cache_data      : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """List label files."""
        files = self.annotation_files()
        if not (is_same_length(files, self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of `files` and `labels` must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        labels: list[VOCLabel] = []
        with progress_bar() as pbar:
            for f in pbar.track(
                files, description=f"[red]Listing {self.split} labels"
            ):
                labels.append(
                    VOCLabel.create_from_file(
                        path         = f,
                        class_labels = self.class_label
                    )
                )
        
        self.labels = labels
        
    @abstractmethod
    def annotation_files(self) -> Paths_:
        """
        Returns the path to json annotation files.
        """
        pass


# H2: - Enhancement ------------------------------------------------------------

class ImageEnhancementDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base type for datasets that represent a collection of images and a set
    of associated enhanced images.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
		Args:
            index (int): The index of the sample to be retrieved.

		Returns:
			input (Tensor[1, C, H, W]): Input sample, optionally transformed by
			    the respective transforms.
			target (Tensor[1, C, H, W]): Enhance image, optionally transformed
			    by the respective transforms.
			meta (Image): Metadata of image.
		"""
        input  = self.images[index].image
        target = self.labels[index].image
        input  = self.images[index].load() if input  is None else input
        target = self.labels[index].load() if target is None else target
        meta   = self.images[index].meta
        
        if self.transform is not None:
            input, *_  = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"[red]Caching {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        return input, target, meta
    

# H2: - Segmentation -----------------------------------------------------------

class ImageSegmentationDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of images and a set
    of associated semantic segmentations.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """
        Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
		Args:
            index (int): The index of the sample to be retrieved.
          
		Returns:
			input (Tensor[1, C, H, W]): Input sample, optionally transformed by
			    the respective transforms.
			target (Tensor[1, C or 1, H, W]): Semantic segmentation mask,
			    optionally transformed by the respective transforms.
			meta (Image): Metadata of image.
		"""
        input  = self.images[index].image
        target = self.labels[index].image
        input  = self.images[index].load() if input  is None else input
        target = self.labels[index].load() if target is None else target
        meta   = self.images[index].meta

        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
    
    def cache_images(self):
        """
        Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"[red]Caching {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"[red]Caching {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """
        Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        return input, target, meta


# H2: - Multitask --------------------------------------------------------------

class ImageLabelsDataset(LabeledImageDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of images and a set
    of associated multitask predictions.
    """
    pass


class VideoLabelsDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """
    Base class for datasets that represent a collection of videos and a set
    of associated multitask predictions.
    """
    pass


# H1: - Device -----------------------------------------------------------------

def extract_device_dtype(tensors: list) -> tuple[torch.device, torch.dtype]:
    """
    Take a list of tensors and returns a tuple of `device` and `dtype` that
    are the same for all tensors in the list.
    
    Args:
        tensors (list): list.
    
    Returns:
        A tuple of the device and dtype of the tensor.
    """
    device, dtype = None, None
    for tensors in tensors:
        if tensors is not None:
            if not isinstance(tensors, (Tensor,)):
                continue
            _device = tensors.device
            _dtype  = tensors.dtype
            if device is None and dtype is None:
                device = _device
                dtype  = _dtype
            
            if device != _device or dtype != _dtype:
                raise ValueError(
                    f"Passed values must be in the same `device` and `dtype`. "
                    f"But got: ({device}, {dtype}) and ({_device}, {_dtype})."
                )
                
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype  = torch.get_default_dtype()
    return device, dtype


def get_gpu_memory(
    device_index: int = 0, unit: MemoryUnit = MemoryUnit.GB
) -> tuple[int, int, int]:
    """
    Return the total, used, and free memory of the GPU with the given index
    in the given unit.
    
    Args:
        device_index (int): The index of the GPU you want to get the memory
            usage of. Defaults to 0.
        unit (MemoryUnit_): MemoryUnit_ = MemoryUnit.GB.
    
    Returns:
        A tuple of the total, used, and free memory in the specified unit.
    """
    nvmlInit()
    unit  = MemoryUnit.from_value(unit)
    h     = nvmlDeviceGetHandleByIndex(device_index)
    info  = nvmlDeviceGetMemoryInfo(h)
    ratio = MemoryUnit.byte_conversion_mapping()[unit]
    total = info.total / ratio
    free  = info.free  / ratio
    used  = info.used  / ratio
    return total, used, free


def select_device(
    device    : str | None = "",
    batch_size: int | None = None
) -> torch.device:
    """
    Return a torch.device object, which is either cuda:0 or cpu, depending
    on whether CUDA is available.
    
    Args:
        device (str | None): The device to run the model on. If None, the
          default device is used.
        batch_size (int | None): The number of samples to process in a single
            batch.
    
    Returns:
        A torch.device object.
    """
    device      = f"{device}"
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # If device requested other than `cpu`
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        if not torch.cuda.is_available():  # Check availability
            raise RuntimeError(
                f"CUDA unavailable, invalid device {device} requested."
            )
            
    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        bytes_to_mb = 1024 ** 2  # bytes to MB
        num_gpus    = torch.cuda.device_count()
        if num_gpus > 1 and batch_size:  # check that batch_size is compatible with device_count
            if batch_size % num_gpus != 0:
                raise ValueError(
                    f"`batch-size` must be a multiple of GPU count {num_gpus}. "
                    f"But got: {batch_size} % {num_gpus} != 0."
                )
        
        x = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]
        s = "Using CUDA "
        for i in range(0, num_gpus):
            if i == 1:
                s = " " * len(s)
            console.log(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / bytes_to_mb)
            )
    else:
        console.log("Using CPU")

    console.log("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def select_device_old(
    model_name: str        = "",
    device    : str | None = "",
    batch_size: int | None = None
) -> torch.device:
    """
    Set the environment variable `CUDA_VISIBLE_DEVICES` to the requested
    device, and returns a `torch.device` object.
    
    Args:
        model_name (str): The name of the model. This is used to print a message
            to the console.
        device (str | None): The device to run the model on. If None, the
          default device is used.
        batch_size (int | None): The number of samples to process in a single
            batch.
    
    Returns:
        A torch.device object.
    """
    if device is None:
        return torch.device("cpu")

    # device = 'cpu' or '0' or '0,1,2,3'
    s   = f"{model_name}"  # string
    
    if isinstance(device, str) and device.lower() == "cpu":
        cpu = True
    else:
        cpu = False
    
    if cpu:
        # Force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        # Non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # Check availability
        if not torch.cuda.is_available():  # Check availability
            raise RuntimeError(
                f"CUDA unavailable, invalid device {device} requested."
            )
    
    cuda = not cpu and torch.cuda.is_available()
    
    if cuda:
        n = torch.cuda.device_count()

        # Check that batch_size is compatible with device_count
        if n > 1 and batch_size:
            assert_number_divisible_to(batch_size, n)
        space = " " * len(s)
        
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"
    
    console.log(s)
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    """
    Synchronize the CUDA device if it's available, and then returns the
    current time.
    
    Returns:
        The time in seconds since the epoch as a floating point number.
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# H1: - Factory ----------------------------------------------------------------

class Registry:
    """
    Base registry class for registering classes.

    Args:
        name (str):
            Registry name.
    """
    
    def __init__(self, name: str):
        self._name     = name
        self._registry = {}
    
    def __len__(self) -> int:
        """
        The function returns the length of the registry.
        
        Returns:
            The length of the registry.
        """
        return len(self._registry)
    
    def __contains__(self, key: str):
        """
        If the key is in the dictionary, return the value associated with the
        key, otherwise return the default value.
        
        Args:
            key (str): The key to look for.
        
        Returns:
            The value of the key.
        """
        return self.get(key) is not None
    
    def __repr__(self):
        """
        The __repr__ function returns a string that contains the name of the
        class, the name of the object, and the contents of the registry.
        
        Returns:
            The name of the class and the name of the registry.
        """
        format_str = self.__class__.__name__ \
                     + f"(name={self._name}, items={self._registry})"
        return format_str
        
    @property
    def name(self) -> str:
        """
        It returns the name of the object.
        
        Returns:
            The name of the registry.
        """
        return self._name
    
    @property
    def registry(self) -> dict:
        """
        It returns the dictionary of the class.
        
        Returns:
            A dictionary.
        """
        return self._registry
    
    def get(self, key: str) -> Callable:
        """
        If the key is in the registry, return the value.
        
        Args:
            key (str): The name of the command.
        
        Returns:
            A callable object.
        """
        if key in self._registry:
            return self._registry[key]
        
    def register(
        self,
        name  : str | None = None,
        module: Callable   = None,
        force : bool       = False
    ) -> callable:
        """
        It can be used as a normal method or as a decorator.
        
        Example:
            # >>> backbones = Factory("backbone")
            # >>>
            # >>> @backbones.register()
            # >>> class ResNet:
            # >>>     pass
            # >>>
            # >>> @backbones.register(name="mnet")
            # >>> class MobileNet:
            # >>>     pass
            # >>>
            # >>> class ResNet:
            # >>>     pass
            # >>> backbones.register(ResNet)
            
        Args:
            name (str | None): The name of the module. If not specified, it
                will be the class name.
            module (Callable): The module to register.
            force (bool): If True, it will overwrite the existing module with
                the same name. Defaults to False.
        
        Returns:
            A function.
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                f"`name` must be `None` or `str`. But got: {type(name)}."
            )
        
        # Use it as a normal method: x.register(module=SomeClass)
        if module is not None:
            self.register_module(module, name, force)
            return module
        
        # Use it as a decorator: @x.register()
        def _register(cls):
            self.register_module(cls, name, force)
            return cls
        
        return _register
    
    def register_module(
        self,
        module_class: Callable,
        module_name : str | None = None,
        force       : bool       = False
    ):
        """
        It takes a class and a name, and adds the class to the registry under
        the name.
        
        Args:
            module_class (Callable): The class of the module to be registered.
            module_name (str | None): The name of the module. If not provided,
                it will be the class name in lowercase.
            force (bool): If True, the module will be registered even if it's
                already registered. Defaults to False.
        """
        assert_class(module_class)
        
        if module_name is None:
            module_name = module_class.__name__.lower()
        module_name = to_list(module_name)
        
        for name in module_name:
            if not force and name in self._registry:
                continue
                # logger.debug(f"{name} is already registered in {self.name}.")
            else:
                self._registry[name] = module_class
    
    def print(self):
        """
        It prints the name of the object, and then prints the registry.
        """
        console.log(f"[red]{self.name}:")
        print_table(self.registry)


class Factory(Registry):
    """
    Default factory class for creating objects.
    
    Registered object could be built from registry.
    Example:
        >>> MODELS = Factory("models")
        >>> @MODELS.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODELS.build(name="ResNet", **resnet_hparams)
    """
    
    def build(self, name: str, *args, **kwargs) -> object:
        """
        It takes a name, and returns an instance of the class that is
        registered under that name.
        
        Args:
            name (str): The name of the class to be built.
        
        Returns:
            An instance of the class that is registered with the name.
        """
        assert_dict_contain_key(self.registry, name)
        instance = self.registry[name](*args, **kwargs)
        if not hasattr(instance, "name"):
            instance.name = name
        return instance
    
    def build_from_dict(self, cfg: dict | None, **kwargs) -> object | None:
        """
        Factory command to create a class' instance with arguments given in
        `cfgs`.
        
        Args:
            cfg (dict | None): Class's arguments.
        
        Returns:
            Class's instance.
        """
        if cfg is None:
            return None
        assert_dict(cfg)
        assert_dict_contain_key(cfg, "name")
    
        cfg_  = deepcopy(cfg)
        name  = cfg_.pop("name")
        cfg_ |= kwargs
        return self.build(name=name, **cfg_)
    
    def build_from_dictlist(
        self,
        cfgs: list[dict] | None,
        **kwargs
    ) -> list[object] | None:
        """
        Factory command to create classes' instances with arguments given in
        `cfgs`.

        Args:
            cfgs (list[dict] | None): List of classes' arguments.

        Returns:
            Classes' instances.
        """
        if cfgs is None:
            return None
        assert_list_of(cfgs, dict)
        
        cfgs_     = deepcopy(cfgs)
        instances = []
        for cfg in cfgs_:
            name  = cfg.pop("name")
            cfg  |= kwargs
            instances.append(self.build(name=name, **cfg))
        
        return instances if len(instances) > 0 else None


class OptimizerFactory(Registry):
    """
    Factory class for creating optimizers.
    """
    
    def build(
        self,
        net : nn.Module,
        name: str,
        *args, **kwargs
    ) -> Optimizer | None:
        """
        Factory command to create an optimizer with arguments given in
        `kwargs`.
        
        Args:
            net (nn.Module): Neural network module.
            name (str): Optimizer's name.
        
        Returns:
            Optimizer.
        """
        assert_dict_contain_key(self.registry, name)
        return self.registry[name](params=net.parameters(), *args, **kwargs)
    
    def build_from_dict(
        self,
        net: nn.Module,
        cfg: dict | None,
        **kwargs
    ) -> Optimizer | None:
        """
        Factory command to create an optimizer with arguments given in
        `cfgs`.

        Args:
            net (nn.Module): Neural network module.
            cfg (dict | None): Optimizer's arguments.

        Returns:
            Optimizer.
        """
        if cfg is None:
            return None
        assert_dict(cfg)
        assert_dict_contain_key(cfg, "name")
        
        cfg_  = deepcopy(cfg)
        name  = cfg_.pop("name")
        cfg_ |= kwargs
        return self.build(net=net, name=name, **cfg_)
    
    def build_from_dictlist(
        self,
        net : nn.Module,
        cfgs: list[dict] | None,
        **kwargs
    ) -> list[Optimizer] | None:
        """
        Factory command to create optimizers with arguments given in `cfgs`.

        Args:
            net (nn.Module): List of neural network modules.
            cfgs (list[dict] | None): List of optimizers' arguments.

        Returns:
            Optimizers.
        """
        if cfgs is None:
            return None
        assert_list_of(cfgs, dict)
        
        cfgs_      = deepcopy(cfgs)
        optimizers = []
        for cfg in cfgs_:
            name  = cfg.pop("name")
            cfg  |= kwargs
            optimizers.append(self.build(net=net, name=name, **cfg))
        
        return optimizers if len(optimizers) > 0 else None
    
    def build_from_list(
        self,
        nets: list[nn.Module],
        cfgs: list[dict] | None,
        **kwargs
    ) -> list[Optimizer] | None:
        """
        Factory command to create optimizers with arguments given in `cfgs`.

        Args:
            nets (list[nn.Module]): List of neural network modules.
            cfgs (list[dict] | None): List of optimizers' arguments.

        Returns:
            Optimizers.
        """
        if cfgs is None:
            return None
        assert_list_of(cfgs, dict)
        assert_list_of(nets, item_type=dict)
        assert_same_length(nets, cfgs)
     
        cfgs_      = deepcopy(cfgs)
        optimizers = []
        for net, cfg in zip(nets, cfgs_):
            name  = cfg.pop("name")
            cfg  |= kwargs
            optimizers.append(self.build(net=net, name=name, **cfg))
        
        return optimizers if len(optimizers) > 0 else None


class SchedulerFactory(Registry):
    """
    Factory class for creating schedulers.
    """
    
    def build(
        self,
        optimizer: Optimizer,
        name     : str | None,
        *args, **kwargs
    ) -> _LRScheduler | None:
        """
        Factory command to create a scheduler with arguments given in
        `kwargs`.
        
        Args:
            optimizer (Optimizer): Optimizer.
            name (str | None): Scheduler's name.
        
        Returns:
            Scheduler.
        """
        if name is None:
            return None
        assert_dict_contain_key(self.registry, name)
    
        if name in ["gradual_warmup_scheduler"]:
            after_scheduler = kwargs.pop("after_scheduler")
            if isinstance(after_scheduler, dict):
                name_ = after_scheduler.pop("name")
                if name_ in self.registry:
                    after_scheduler = self.registry[name_](
                        optimizer=optimizer, **after_scheduler
                    )
                else:
                    after_scheduler = None
            return self.registry[name](
                optimizer       = optimizer,
                after_scheduler = after_scheduler,
                *args, **kwargs
            )
        
        return self.registry[name](optimizer=optimizer, *args, **kwargs)
    
    def build_from_dict(
        self,
        optimizer: Optimizer,
        cfg      : dict | Munch | None,
        **kwargs
    ) -> _LRScheduler | None:
        """
        Factory command to create a scheduler with arguments given in `cfg`.

        Args:
            optimizer (Optimizer): Optimizer.
            cfg (dict | None): Scheduler's arguments.

        Returns:
            Scheduler.
        """
        if cfg is None:
            return None
        assert_dict(cfg)
        assert_dict_contain_key(cfg, "name")
    
        cfg_  = deepcopy(cfg)
        name  = cfg_.pop("name")
        cfg_ |= kwargs
        return self.build(optimizer=optimizer, name=name, **cfg_)
    
    def build_from_dictlist(
        self,
        optimizer: Optimizer,
        cfgs     : dict | None,
        **kwargs
    ) -> list[_LRScheduler] | None:
        """
        Factory command to create schedulers with arguments given in `cfgs`.

        Args:
            optimizer (Optimizer): Optimizer.
            cfgs (list[dict] | None): List of schedulers' arguments.

        Returns:
            Schedulers.
        """
        if cfgs is None:
            return None
        assert_list_of(cfgs, dict)
        
        cfgs_      = deepcopy(cfgs)
        schedulers = []
        for cfg in cfgs_:
            name  = cfg.pop("name")
            cfg  |= kwargs
            schedulers.append(self.build(optimizer=optimizer, name=name, **cfg))
        
        return schedulers if len(schedulers) > 0 else None
    
    def build_from_list(
        self,
        optimizers: list[Optimizer],
        cfgs      : list[list[dict]] | None,
        **kwargs
    ) -> list[_LRScheduler] | None:
        """
        Factory command to create schedulers with arguments given in `cfgs`.

        Args:
            optimizers (list[Optimizer]): List of optimizers.
            cfgs (list[list[dict]] | None): 2D-list of schedulers' arguments.

        Returns:
            Schedulers.
        """
        if cfgs is None:
            return None
        if not (is_list_of(cfgs, item_type=list) or
                all(is_list_of(cfg, item_type=dict) for cfg in cfgs)):
            raise TypeError(
                f"`cfgs` must be a 2D `list[dict]`. But got: {type(cfgs)}."
            )
        assert_same_length(optimizers, cfgs)
        
        cfgs_      = deepcopy(cfgs)
        schedulers = []
        for optimizer, cfgs in zip(optimizers, cfgs_):
            for cfg in cfgs:
                name  = cfg.pop("name")
                cfg  |= kwargs
                schedulers.append(
                    self.build(optimizer=optimizer, name=name, **cfg)
                )
        
        return schedulers if len(schedulers) > 0 else None


# NN Layers
ACT_LAYERS           = Factory(name="act_layers")
ATTN_LAYERS          = Factory(name="attn_layers")
ATTN_POOL_LAYERS     = Factory(name="attn_pool_layers")
BOTTLENECK_LAYERS    = Factory(name="bottleneck_layers")
CONV_LAYERS          = Factory(name="conv_layers")
CONV_ACT_LAYERS      = Factory(name="conv_act_layers")
CONV_NORM_ACT_LAYERS = Factory(name="conv_norm_act_layers")
DROP_LAYERS          = Factory(name="drop_layers")
EMBED_LAYERS         = Factory(name="embed_layers")
HEADS 	             = Factory(name="heads")
LINEAR_LAYERS        = Factory(name="linear_layers")
MLP_LAYERS           = Factory(name="mlp_layers")
NORM_LAYERS          = Factory(name="norm_layers")
NORM_ACT_LAYERS      = Factory(name="norm_act_layers")
PADDING_LAYERS       = Factory(name="padding_layers")
PLUGIN_LAYERS        = Factory(name="plugin_layers")
POOL_LAYERS          = Factory(name="pool_layers")
RESIDUAL_BLOCKS      = Factory(name="residual_blocks")
SAMPLING_LAYERS      = Factory(name="sampling_layers")
# Models
BACKBONES            = Factory(name="backbones")
CALLBACKS            = Factory(name="callbacks")
LOGGERS              = Factory(name="loggers")
LOSSES               = Factory(name="losses")
METRICS              = Factory(name="metrics")
MODELS               = Factory(name="models")
MODULE_WRAPPERS      = Factory(name="module_wrappers")
NECKS 	             = Factory(name="necks")
OPTIMIZERS           = OptimizerFactory(name="optimizers")
SCHEDULERS           = SchedulerFactory(name="schedulers")
# Misc
AUGMENTS             = Factory(name="augments")
DATAMODULES          = Factory(name="datamodules")
DATASETS             = Factory(name="datasets")
DISTANCES            = Factory(name="distance_functions")
DISTANCE_FUNCS       = Factory(name="distance_functions")
FILE_HANDLERS        = Factory(name="file_handler")
LABEL_HANDLERS       = Factory(name="label_handlers")
MOTIONS              = Factory(name="motions")
TRANSFORMS           = Factory(name="transforms")


# H1: - File -------------------------------------------------------------------

def copy_file_to(file: Path_, dst: Path_):
    """
    Copy a file to a destination directory.
    
    Args:
        file (str): The path to the file.
        dst (str): The destination directory where the file will be copied to.
    """
    create_dirs(paths=[dst])
    shutil.copyfile(file, dst / file.name)


def create_dirs(paths: Paths_, recreate: bool = False):
    """
    Create a list of directories, if they don't exist.
    
    Args:
        paths (Paths_): A list of paths to create.
        recreate (bool): If True, the directory will be deleted and recreated.
            Defaults to False.
    """
    paths = to_list(paths)
    paths = [Path(p) for p in paths if p is not None]
    paths = unique(paths)
    try:
        for path in paths:
            if is_dir(path) and recreate:
                shutil.rmtree(path)
            if not is_dir(path):
                path.mkdir(parents=True, exist_ok=recreate)
        return 0
    except Exception as err:
        console.log(f"Cannot create directory: {err}.")
    

def delete_files(
    files    : Paths_ = "",
    dirs     : Paths_ = "",
    extension: str    = "",
    recursive: bool   = True
):
    """
    Deletes files.
    
    Args:
        files (Paths_): A list of files to delete.
        dirs (Paths_): A list of directories to search for files to delete.
        extension (str): File extension. Defaults to "".
        recursive (bool): If True, then the function will search for files
            recursively. Defaults to True.
    """
    files     = to_list(files)
    files     = [Path(f) for f in files if f is not None]
    files     = [f for f in files if f.is_file()]
    files     = unique(files)
    dirs      = to_list(dirs)
    dirs      = [Path(d) for d in dirs if d is not None]
    dirs      = unique(dirs)
    extension = f".{extension}" if "." not in extension else extension
    for d in dirs:
        files += list(d.rglob(*{extension}))
    for f in files:
        console.log(f"Deleting {f}.")
        f.unlink()


def get_hash(files: Paths_) -> int:
    """
    It returns the sum of the sizes of the files in the list.
    
    Args:
        files (Paths_): File paths.
    
    Returns:
        The sum of the sizes of the files in the list.
    """
    files = to_list(files)
    files = [Path(f) for f in files if f is not None]
    return sum(f.stat().st_size for f in files if f.is_file())


def get_latest_file(path: Path_, recursive: bool = True) -> Path | None:
    """
    It returns the latest file in a given directory.
    
    Args:
        path (Path_): The path to the directory you want to search.
        recursive (bool): If True, the pattern “**” will match any files and
            zero or more directories and subdirectories. Defaults to True
    
    Returns:
        The latest file in the path.
    """
    if path is not None:
        file_list = list(Path(path).rglob("*"))
        if len(file_list) > 0:
            return max(file_list, key=os.path.getctime)
    return None


def has_subdir(path: Path_, name: str) -> bool:
    """
    Return True if the directory at the given path has a subdirectory with the
    given name.
    
    Args:
        path (Path_): The path to the directory you want to check.
        name (str): The name of the subdirectory to check for.
    
    Returns:
        A boolean value.
    """
    return name in list_subdirs(path)


def list_files(patterns: Strs) -> list[Path_]:
    """
    It takes a list of file patterns and returns a list of all files that match
    those patterns.
    
    Args:
      patterns (Strs): A list of file paths to search for images.
    
    Returns:
        A list of unique file paths.
    """
    patterns    = to_list(patterns)
    patterns    = [p for p in patterns if p is not None]
    patterns    = unique(patterns)
    image_paths = []
    for pattern in patterns:
        for abs_path in glob.glob(pattern):
            if os.path.isfile(abs_path):
                image_paths.append(Path(abs_path))
    return unique(image_paths)


def list_subdirs(path: Path_ | None) -> list[Paths_] | None:
    """
    It returns a list of all the subdirectories of the given path
    
    Args:
        path (Path_ | None): The given path.
    
    Returns:
        A list of all the subdirectories in the given path.
    """
    if path is None:
        return None
    path = str(Path)
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


# H1: - Logging ----------------------------------------------------------------

field_style = {
    "asctime"  : {"color": "green"},
    "levelname": {"bold" : True},
    "filename" : {"color": "cyan"},
    "funcName" : {"color": "blue"}
}

level_styles = {
    "critical": {"bold" : True, "color": "red"},
    "debug"   : {"color": "green"},
    "error"   : {"color": "red"},
    "info"    : {"color": "magenta"},
    "warning" : {"color": "yellow"}
}

rich_console_theme = Theme(
    {
        "debug"   : "dark_green",
        "info"    : "green",
        "warning" : "yellow",
        "error"   : "bright_red",
        "critical": "bold red",
    }
)

console = Console(
    color_system    = "windows",
    log_time_format = "[%x %H:%M:%S:%f]",
    soft_wrap       = True,
    theme           = rich_console_theme,
)

error_console = Console(
    color_system    = "windows",
    log_time_format = "[%x %H:%M:%S:%f]",
    soft_wrap       = True,
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)


def download_bar() -> Progress:
    """
    It returns a progress bar that displays the current time, the task
    description, a progress bar, the percentage complete, the transfer speed,
    the amount downloaded, the time remaining, the time elapsed, and a
    right-pointing arrow.
    
    Returns:
        A progress bar.
    """
    return Progress(
        TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
            justify="left", style="log.time"
        ),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
        ">",
        TimeElapsedColumn(),
        console=console,
    )


def get_logger(log_file: Path_ | None = None):
    """
    It creates a logger that logs to a file if a file is provided.
    
    Args:
        log_file (Path_ | None): The given log file.
    
    Returns:
        A logger object.
    """
    if log_file:
        file = logging.FileHandler(log_file)
        file.setLevel(logging.INFO)
        file.setFormatter(logging.Formatter(
            " %(asctime)s [%(filename)s %(lineno)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(file)

    return logger


def progress_bar() -> Progress:
    """
    It returns a progress bar that displays the current time, the task
    description, a progress bar, the percentage complete, the number of items
    processed, the processing speed, the time remaining, the time elapsed,
    and a spinner.
    
    Returns:
        A progress bar.
    """
    return Progress(
        TextColumn(
            console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
            justify="left", style="log.time"
        ),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None, finished_style="green"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        ProcessedItemsColumn(),
        "•",
        ProcessingSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        ">",
        TimeElapsedColumn(),
        SpinnerColumn(),
        console=console,
    )


def print_dict(data: dict, title: str = ""):
    """
    It takes a dictionary and prints it in a pretty format.
    
    Args:
        data (dict): The data to be printed.
        title (str): The title of the panel.
    """
    if isinstance(data, Munch):
        data = data.toDict()
        
    pretty = Pretty(
        data,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    panel = Panel(pretty, title=f"{title}")
    console.log(panel)
    

@dispatch(list)
def print_table(data: list[dict]):
    """
    This function takes a list of dictionaries and prints them in a table.
    
    Args:
        data (list[dict]): A list of dictionary.
    """
    assert_list_of(data, dict)
    table = Table(show_header=True, header_style="bold magenta")
    for k, v in data[0].items():
        table.add_column(k)
    
    for d in data:
        row = [f"{v}" for v in d.values()]
        table.add_row(*row)
    
    console.log(table)


@dispatch(dict)
def print_table(data: dict):
    """
    It takes a dictionary and prints it as a table.
    
    Args:
        data (dict): dict
    """
    assert_dict(data)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Key")
    table.add_column("Value")

    for k, v in data.items():
        row = [f"{k}", f"{v}"]
        table.add_row(*row)

    console.log(table)


class GPUMemoryUsageColumn(ProgressColumn):
    """
    Renders GPU memory usage, e.g. `33.1/48.0GB`.
    
    Args:
        unit (MemoryUnit_): The unit of memory to use.
        table_column (Column | None): The column in the table that this field
            is associated with.
    """

    def __init__(
        self,
        unit        : MemoryUnit    = MemoryUnit.GB,
        table_column: Column | None = None
    ):
        super().__init__(table_column=table_column)
        self.unit = MemoryUnit.from_value(unit)

    def render(self, task: Task) -> Text:
        """
        It returns a Text object with the memory usage of the GPU.
        
        Args:
            task (Task): Task.
        
        Returns:
            A Text object with the memory status.
        """
        total, used, free = get_gpu_memory()
        memory_status     = f"{used:.1f}/{total:.1f}{self.unit.value}"
        memory_text       = Text(memory_status, style="bright_yellow")
        return memory_text
    

class ProcessedItemsColumn(ProgressColumn):
    """
    Renders processed files and total, e.g. `1728/2025`.
    
    Args:
        table_column (Column | None): The column that this widget is
            associated with.
    """

    def __init__(self, table_column: Column | None = None):
        super().__init__(table_column=table_column)

    def render(self, task: Task) -> Text:
        """
        It takes a Task object and returns a Text object.
        
        Args:
            task (Task): Task.
        
        Returns:
            A Text object with the download status.
        """
        completed       = int(task.completed)
        total           = int(task.total)
        download_status = f"{completed}/{total}"
        download_text   = Text(download_status, style="progress.download")
        return download_text


class ProcessingSpeedColumn(ProgressColumn):
    """
    Renders human-readable progress speed.
    """

    def render(self, task: Task) -> Text:
        """
        It takes a task and returns a Text object.
        
        Args:
            task (Task): Task.
        
        Returns:
            A Text object with the speed data.
        """
        speed = task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        speed_data = "{:.2f}".format(speed)
        return Text(f"{speed_data}it/s", style="progress.data.speed")
    
    
# H1: - Serialization ----------------------------------------------------------

def dump_to_file(
    obj        : Any,
    path       : Path_,
    file_format: str | None = None,
    **kwargs
) -> bool | str:
    """
    It dumps an object to a file or a file-like object.
    
    Args:
        obj (Any): The object to be dumped.
        path (Path_): The path to the file to be written.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        A boolean or a string.
    """
    path = Path(path)
    if file_format is None:
        file_format = path.suffix
    assert_dict_contain_key(FILE_HANDLERS, file_format)
    
    handler: BaseFileHandler = FILE_HANDLERS.build(name=file_format)
    if path is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(path, str):
        handler.dump_to_file(obj, path, **kwargs)
    elif hasattr(path, "write"):
        handler.dump_to_fileobj(obj, path, **kwargs)
    else:
        raise TypeError("`path` must be a filename str or a file-object.")


def load_config(config: str | dict | Munch) -> Munch:
    """
    Load config as namespace.

	Args:
		config (str | dict | Munch): Config filepath that contains
		configuration values
		    or the config dict.
	"""
    # Load dictionary from file and convert to namespace using Munch
    if not isinstance(config, (str, dict, Munch)):
        raise TypeError(
            f"`config` must be a `dict` or a path to config file. "
            f"But got: {config}."
        )
    if isinstance(config, str):
        config_dict = load_from_file(path=config)
    else:
        config_dict = config
    
    if config_dict is None:
        raise IOError(f"No configuration is found at: {config}.")
    
    config = Munch.fromDict(config_dict)
    return config


def load_from_file(
    path       : Path_,
    file_format: str | None = None,
    **kwargs
) -> str | dict | None:
    """
    Load a file from a filepath or file-object, and return the data in the file.
    
    Args:
        path (Path_): The path to the file to load.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        The data from the file.
    """
    path = Path(path)
    if file_format is None:
        file_format = path.suffix
    assert_dict_contain_key(FILE_HANDLERS, file_format)

    handler: BaseFileHandler = FILE_HANDLERS.build(name=file_format)
    if isinstance(path, str):
        data = handler.load_from_file(path, **kwargs)
    elif hasattr(path, "read"):
        data = handler.load_from_fileobj(path, **kwargs)
    else:
        raise TypeError("`file` must be a filepath str or a file-object.")
    return data


def merge_files(
    in_paths   : Paths_,
    out_path   : Path_,
    file_format: str | None = None,
) -> bool | str:
    """
    Reads data from multiple files and writes it to a single file.
    
    Args:
        in_paths (Paths_): The input paths to the files you want to merge.
        out_path (Path_): The path to the output file.
        file_format (str | None): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently, supported formats include "json", "yaml/yml" and
            "pickle/pkl". Default: `None`.
    
    Returns:
        A boolean or a string.
    """
    in_paths = to_list(in_paths)
    in_paths = [Path(p) for p in in_paths]
    
    # Read data
    data = None
    for p in in_paths:
        d = load_from_file(p)
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
    return dump_to_file(obj=data, path=out_path, file_format=file_format)


class BaseFileHandler(metaclass=ABCMeta):
    """
    Base file handler implements the template methods (i.e., skeleton) for
    read and write data from/to different file formats.
    """
    
    @abstractmethod
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        It loads a file from a file object.
        
        Args:
            path (Path_): The path to the file to load.
        """
        pass
        
    @abstractmethod
    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It takes a `self` object, an `obj` object, a `path` object, and a
        `**kwargs` object, and returns nothing.
        
        Args:
            obj: The object to be dumped.
            path (Path_): The path to the file to be read.
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
        self, path: Path_, mode: str = "r", **kwargs
    ) -> str | dict | None:
        """
        It loads a file from the given path and returns the contents.
        
        Args:
            path (Path_): The path to the file to load from.
            mode (str): The mode to open the file in. Defaults to "r".
        
        Returns:
            The return type is a string, dictionary, or None.
        """
        with open(path, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_file(self, obj, path: Path_, mode: str = "w", **kwargs):
        """
        It writes the object to a file.
        
        Args:
            obj: The object to be serialized.
            path (Path): The path to the file to write to.
            mode (str): The mode in which the file is opened. Defaults to "w".
        """
        with open(path, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


@FILE_HANDLERS.register(name="json")
class JsonHandler(BaseFileHandler):
    """
    JSON file handler.
    """
    
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
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        This function loads a json file from a file object and returns a
        string, dictionary, or None.
        
        Args:
            path (Path_): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        return json.load(path)

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It dumps the object to a file object.
        
        Args:
            obj: The object to be serialized.
            path (Path_): The path to the file to write to.
        """
        path = Path(path)
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
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        This function loads a pickle file from a file object.
        
        Args:
            path (Path_): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        return pickle.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        Takes a Python object, a path to a file, and a set of keyword arguments,
        and writes the object to the file using the pickle module.
        
        Args:
            obj: The object to be pickled.
            path (Path_): The path to the file to be opened.
        """
        path = Path(path)
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
        
    def load_from_file(self, path: Path_, **kwargs) -> str | dict | None:
        """
        Loads a file from the file system and returns the contents as a string,
        dictionary, or None.
        
        Args:
            path (Path_): Path: The file to load from.
        
        Returns:
            The return value is a string or a dictionary.
        """
        path = Path(path)
        return super().load_from_file(path, mode="rb", **kwargs)
    
    def dump_to_file(self, obj, path: Path_, **kwargs):
        """
        It dumps the object to a file.
        
        Args:
            obj: The object to be serialized.
            path (Path_): The path to the file to which the object is to be
                dumped.
        """
        path = Path(path)
        super().dump_to_file(obj, path, mode="wb", **kwargs)


@FILE_HANDLERS.register(name="xml")
class XmlHandler(BaseFileHandler):
    """
    XML file handler.
    """
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        It takes a path to a file, reads the file, parses the XML, and returns a
        dictionary.
        
        Args:
            path (Path_): The path to the file to load from.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        doc = xmltodict.parse(path.read())
        return doc

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It takes a dictionary, converts it to XML, and writes it to a file.
        
        Args:
            obj: The object to be dumped.
            path (Path_): The path to the file to be read.
        """
        path = Path(path)
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
    
    def load_from_fileobj(self, path: Path_, **kwargs) -> str | dict | None:
        """
        It loads a YAML file from a file object.
        
        Args:
            path (Path): The path to the file to load.
        
        Returns:
            The return value is a string, dictionary, or None.
        """
        path = Path(path)
        kwargs.setdefault("Loader", FullLoader)
        return yaml.load(path, **kwargs)

    def dump_to_fileobj(self, obj, path: Path_, **kwargs):
        """
        It takes a Python object, a path to a file, and a set of keyword
        arguments, and writes the object to the file using the `Dumper` class.
        
        Args:
            obj: The Python object to be serialized.
            path (Path): The file object to dump to.
        """
        path = Path(path)
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


# H1: - Transform  -------------------------------------------------------------

class Transform(nn.Module, ABC):
    """
    Transform module.
    
    Args:
        p (float | None): Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__()
        if p is not None:
            assert_number_in_range(p, 0.0, 1.0)
        self.p = p
    
    def __repr__(self) -> str:
        """
        The `__repr__` function returns a string representation of the object.
        
        Returns:
            The class name.
        """
        return f"{self.__class__.__name__}()"
    
    def __call__(
        self,
        input  : Tensor,
        target : Tensor  | None = None,
        dataset: Dataset | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        If the probability is greater than a random number, then apply
        transforms, otherwise return the input.
        
        Args:
            input (Tensor): The input tensor to be transformed.
            target (Tensor | None): The target tensor to be transformed.
                Defaults to None.
            dataset (Dataset | None): The dataset. Defaults to None.
            
        Returns:
            The input and target tensors.
        """
        if self.p is None or torch.rand(1).item() <= self.p:
            return super().__call__(
                input   = input,
                target  = target,
                dataset = dataset,
                *args, **kwargs
            )
        return input, target
        
    @abstractmethod
    def forward(
        self,
        input  : Tensor,
        target : Tensor  | None = None,
        dataset: Dataset | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        The function `forward` takes a tensor `input` and an optional tensor
        `target` and returns a tuple of two tensors, the first being the
        transformed input and the second being the transformed target.
        
        Args:
            input (Tensor):  The input tensor to be transformed.
            target (Tensor | None): The target tensor to be transformed.
                Defaults to None.
            dataset (Dataset | None): The dataset. Defaults to None.
        
        Returns:
            The input and target tensors.
        """
        pass


class ComposeTransform(nn.Sequential):
    """
    Composes several transforms together. This transform support torchscript.
    Please, see the note below.

    Args:
        transforms (ScalarOrCollectionT[Transform | dict]):
            List of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    Notes:
        In order to script the transformations, please use
        `torch.nn.Sequential` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with
        `torch.Tensor`, does not require `lambda` functions or `PIL.Image`.
    """
    
    def __init__(
        self,
        transforms: ScalarOrCollectionT[Transform | dict],
        *args, **kwargs
    ):
        if isinstance(transforms, dict):
            transforms = [v for k, v in transforms.items()]
        if isinstance(transforms, list):
            transforms = [
                TRANSFORMS.build_from_dict(cfg=t)
                if isinstance(t, dict) else t
                for t in transforms
            ]
            if not all(isinstance(t, Transform) for t in transforms):
                raise TypeError(f"All items in `transforms` must be callable.")
        
        args = transforms + list(args)
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        input  : Tensor | np.ndarray | PIL.Image,
        target : Tensor | np.ndarray | PIL.Image | None = None,
        dataset: Dataset | None                         = None,
        *args, **kwargs
    ) -> tuple[
        Tensor | np.ndarray | PIL.Image,
        Tensor | np.ndarray | PIL.Image | None
    ]:
        """
        It applies the transforms to the input and target.
        
        Args:
            input (Tensor | np.ndarray | PIL.Image): The input tensor to be
                transformed.
            target (Tensor | np.ndarray | PIL.Image | None): The target tensor
                to be transformed.
            dataset (Dataset | None): The dataset. Defaults to None.
            
        Returns:
            The transformed input and target.
        """
        # for t in self.named_modules():
        #     input, target = t(input, target)
        return super().__call__(
            input   = input,
            target  = target,
            dataset = dataset,
            *args, **kwargs
        )
    
    def __repr__(self) -> str:
        """
        The function returns a string that contains the name of the class,
        and the string representation of each transform in the list of
        transforms.
        
        Returns:
            A string representation of the object.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
    def forward(
        self,
        input  : Tensor | np.ndarray | PIL.Image,
        target : Tensor | np.ndarray | PIL.Image | None = None,
        dataset: Dataset | None                         = None,
        *args, **kwargs
    ) -> tuple[
        Tensor | np.ndarray | PIL.Image,
        Tensor | np.ndarray | PIL.Image | None
    ]:
        """
        It applies the transforms to the input and target.
        
        Args:
            input (Tensor | np.ndarray | PIL.Image): The input tensor to be
                transformed.
            target (Tensor | np.ndarray | PIL.Image | None): The target tensor
                to be transformed.
            dataset (Dataset | None): The dataset. Defaults to None.
            
        Returns:
            The transformed input and target.
        """
        for t in self:
            input, target = t(
                input   = input,
                target  = target,
                dataset = dataset,
                *args, **kwargs
            )
        return input, target


# H1: - Typing -----------------------------------------------------------------
# Template
T                   = TypeVar("T")
ScalarOrSequenceT   = Union[T, Sequence[T]]
ScalarOrCollectionT = Union[T, Collection[T]]
# Basic Types
Callable            = Union[typing.Callable, types.FunctionType, functools.partial]
Ints                = ScalarOrSequenceT[int]
Floats              = ScalarOrSequenceT[float]
Numbers             = ScalarOrSequenceT[Number]
Strs                = ScalarOrSequenceT[str]
# Custom Types
Arrays              = ScalarOrCollectionT[np.ndarray]
ClassLabel_         = Union[ClassLabel, str, list, dict]
Color               = Sequence[int]
Colors              = Sequence[Color]
Devices             = Union[Ints, Strs]
Enum_               = Union[Enum, str, int]
Path_               = Union[str, pathlib.Path]
Paths_              = ScalarOrSequenceT[Path_]
TensorOrArray       = Union[Tensor, np.ndarray]
Tensors             = ScalarOrCollectionT[Tensor]
Transforms_         = Union[ScalarOrCollectionT[Union[Transform, dict]],
                            ComposeTransform]
# DataLoader Types
EvalDataLoaders     = ScalarOrSequenceT[DataLoader]
TrainDataLoaders    = Union[DataLoader,
                            Sequence[DataLoader],
                            Sequence[Sequence[DataLoader]],
                            Sequence[dict[str, DataLoader]],
                            dict[str, DataLoader],
                            dict[str, dict[str, DataLoader]],
                            dict[str, Sequence[DataLoader]]]
# Enum Types
InterpolationMode_  = Union[InterpolationMode, str, int]
ModelPhase_         = Union[ModelPhase,        str, int]
PaddingMode_        = Union[PaddingMode,       str, int]
VisionBackend_      = Union[VisionBackend,     str, int]
# Model Building Types
Losses_             = ScalarOrCollectionT[Union[_Loss,     dict]]
Metrics_            = ScalarOrCollectionT[Union[Metric,    dict]]
Optimizers_         = ScalarOrCollectionT[Union[Optimizer, dict]]
Paddings            = Union[Ints, str]
Pretrained          = Union[bool, str, dict]
Weights             = Union[Tensor, Numbers]
