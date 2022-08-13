#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core module that defines all types, classes, and helper functions used
throughout `One` package.

Basically, this is just a one glorified utils package that reduce the
complexity of import statements.

Taxonomy:
    |
    |__ Enum
    |__ Assertion
    |__ Conversion
    |__ Device
    |__ Factory
    |__ File
    |__ Logging
    |__ Transform
    |__ Typing
"""

from __future__ import annotations

import collections
import functools
import glob
import inspect
import itertools
import logging
import os
import pathlib
import random
import shutil
import time
import types
import typing
from abc import ABC
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
import torch
import validators
from multipledispatch import dispatch
from munch import Munch
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


# H1: - Enum -------------------------------------------------------------------

class AppleRGB(Enum):
    """
    Define 12 Apple colors.
    """
    GRAY   = (128, 128, 128)
    RED    = (255,  59,  48)
    GREEN  = ( 52, 199,  89)
    BLUE   = (  0, 122, 255)
    ORANGE = (255, 149,   5)
    YELLOW = (255, 204,   0)
    BROWN  = (162, 132,  94)
    PINK   = (255,  45,  85)
    PURPLE = ( 88,  86, 214)
    TEAL   = ( 90, 200, 250)
    INDIGO = ( 85, 190, 240)
    BLACK  = (  0,   0,   0)
    WHITE  = (255, 255, 255)

    @classmethod
    def random(cls):
        """
        Return a random AppleRGB enum.
        
        Returns:
            A random choice from the list of AppleRGB.
        """
        return random.choice(list(cls))
    
    @classmethod
    def random_value(cls):
        """
        Return a random color.
        
        Returns:
            A random color from the list of AppleRGB.
        """
        return cls.random().value

    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.

        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


class BasicRGB(Enum):
    """
    Define 12 basic colors.
    """
    BLACK   = (  0,   0,   0)
    WHITE   = (255, 255, 255)
    RED     = (255,   0,   0)
    LIME    = (  0, 255,   0)
    BLUE    = (  0,   0, 255)
    YELLOW  = (255, 255,   0)
    CYAN    = (  0, 255, 255)
    MAGENTA = (255,   0, 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    MAROON  = (128,   0,   0)
    OLIVE   = (128, 128,   0)
    GREEN   = (  0, 128,   0)
    PURPLE  = (128,   0, 128)
    TEAL    = (  0, 128, 128)
    NAVY    = (  0,   0, 128)
    
    @classmethod
    def random(cls):
        """
        Return a random BasicRGB value.
        
        Returns:
            A random choice from the list of BasicRGB.
        """
        return random.choice(list(cls))
    
    @classmethod
    def random_value(cls):
        """
        Return a random color.
        
        Returns:
            A random color from the list of BasicRGB.
        """
        return cls.random().value

    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.

        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


class BBoxFormat(Enum):
    CXCYAR      = "cxcyar"
    CXCYRH      = "cxcyrh"
    CXCYWH      = "cxcywh"
    CXCYWH_NORM = "cxcywh_norm"
    XYXY        = "xyxy"
    XYWH        = "xywh"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "cxcyar"     : cls.CXCYAR,
            "cxcyrh"     : cls.CXCYRH,
            "cxcywh"     : cls.CXCYWH,
            "cxcywh_norm": cls.CXCYWH_NORM,
            "xyxy"       : cls.XYXY,
            "xywh"       : cls.XYWH,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.CXCYAR,
            1: cls.CXCYRH,
            2: cls.CXCYWH,
            3: cls.CXCYWH_NORM,
            4: cls.XYXY,
            5: cls.XYWH,
        }
    
    @classmethod
    def from_str(cls, value: str) -> BBoxFormat:
        """
        It takes a string and returns an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BBoxFormat:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> BBoxFormat | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `BBoxFormat`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None
    
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


class BorderType(Enum):
    CONSTANT      = "constant"
    CIRCULAR      = "circular"
    REFLECT       = "reflect"
    REPLICATE     = "replicate"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "constant"   : cls.CONSTANT,
            "circular"   : cls.CIRCULAR,
            "reflect"    : cls.REFLECT,
            "replicate"  : cls.REPLICATE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : cls.CONSTANT,
            1 : cls.CIRCULAR,
            2 : cls.REFLECT,
            3 : cls.REPLICATE,
        }

    @classmethod
    def from_str(cls, value: str) -> BorderType:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value)
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BorderType:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> BorderType | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, BorderType):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `BorderType`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None
        
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


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
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "braycurtis"        : cls.BRAYCURTIS,
            "canberra"          : cls.CANBERRA,
            "chebyshev"         : cls.CHEBYSHEV,
            "cityblock"         : cls.CITYBLOCK,
            "correlation"       : cls.CORRELATION,
            "cosine"            : cls.COSINE,
            "dice"              : cls.DICE,
            "directed_hausdorff": cls.DIRECTED_HAUSDORFF,
            "euclidean"         : cls.EUCLIDEAN,
            "hamming"           : cls.HAMMING,
            "jaccard"           : cls.JACCARD,
            "jensenshannon"     : cls.JENSENSHANNON,
            "kulczynski1"       : cls.KULCZYNSKI1,
            "kulsinski"         : cls.KULSINSKI,
            "mahalanobis"       : cls.MAHALANOBIS,
            "minkowski"         : cls.MINKOWSKI,
            "rogerstanimoto"    : cls.ROGERSTANIMOTO,
            "russellrao"        : cls.RUSSELLRAO,
            "seuclidean"        : cls.SEUCLIDEAN,
            "sokalmichener"     : cls.SOKALMICHENER,
            "sokalsneath"       : cls.SOKALSNEATH,
            "sqeuclidean"       : cls.SQEUCLIDEAN,
            "yule"              : cls.YULE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : cls.BRAYCURTIS,
            1 : cls.CANBERRA,
            2 : cls.CHEBYSHEV,
            3 : cls.CITYBLOCK,
            4 : cls.CORRELATION,
            5 : cls.COSINE,
            6 : cls.DICE,
            7 : cls.DIRECTED_HAUSDORFF,
            8 : cls.EUCLIDEAN,
            9 : cls.HAMMING,
            10: cls.JACCARD,
            11: cls.JENSENSHANNON,
            12: cls.KULCZYNSKI1,
            13: cls.KULSINSKI,
            14: cls.MAHALANOBIS,
            15: cls.MINKOWSKI,
            16: cls.ROGERSTANIMOTO,
            17: cls.RUSSELLRAO,
            18: cls.SEUCLIDEAN,
            19: cls.SOKALMICHENER,
            20: cls.SOKALSNEATH,
            21: cls.SQEUCLIDEAN,
            22: cls.YULE,
        }
    
    @classmethod
    def from_str(cls, value: str) -> DistanceMetric:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> DistanceMetric:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> DistanceMetric | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `DistanceMetric`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None
    
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]
 

class ImageFormat(Enum):
    ARW  = ".arw"
    BMP  = ".bmp"
    DNG	 = ".dng"
    JPG  = ".jpg"
    JPEG = ".jpeg"
    PNG  = ".png"
    PPM  = ".ppm"
    RAF  = ".raf"
    TIF  = ".tif"
    TIFF = ".tiff"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "arw" : cls.ARW,
            "bmp" : cls.BMP,
            "dng" : cls.DNG,
            "jpg" : cls.JPG,
            "jpeg": cls.JPEG,
            "png" : cls.PNG,
            "ppm" : cls.PPM,
            "raf" : cls.RAF,
            "tif" : cls.TIF,
            "tiff": cls.TIF,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.ARW,
            1: cls.BMP,
            2: cls.DNG,
            3: cls.JPG,
            4: cls.JPEG,
            5: cls.PNG,
            6: cls.PPM,
            7: cls.RAF,
            8: cls.TIF,
            9: cls.TIF,
        }
    
    @classmethod
    def from_str(cls, value: str) -> ImageFormat:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ImageFormat:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Enum_) -> ImageFormat | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `ImageFormat`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None
    
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


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

    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "bicubic"      : cls.BICUBIC,
            "bilinear"     : cls.BILINEAR,
            "nearest"      : cls.NEAREST,
            "box"          : cls.BOX,
            "hamming"      : cls.HAMMING,
            "lanczos"      : cls.LANCZOS,
            "area"         : cls.AREA,
            "cubic"        : cls.CUBIC,
            "lanczos4"     : cls.LANCZOS4,
            "linear"       : cls.LINEAR,
            "linear_exact" : cls.LINEAR_EXACT,
            "max"          : cls.MAX,
            "nearest_exact": cls.NEAREST_EXACT,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : cls.BICUBIC,
            1 : cls.BILINEAR,
            2 : cls.NEAREST,
            3 : cls.BOX,
            4 : cls.HAMMING,
            5 : cls.LANCZOS,
            6 : cls.AREA,
            7 : cls.CUBIC,
            8 : cls.LANCZOS4,
            9 : cls.LINEAR,
            10: cls.LINEAR_EXACT,
            11: cls.MAX,
            12: cls.NEAREST_EXACT,
        }

    @classmethod
    def cv_modes_mapping(cls) -> dict:
        """
        It maps the `InterpolationMode` enum to the corresponding OpenCV
        interpolation mode.
        
        Returns:
            A dictionary of the different interpolation modes.
        """
        return {
            cls.AREA    : cv2.INTER_AREA,
            cls.CUBIC   : cv2.INTER_CUBIC,
            cls.LANCZOS4: cv2.INTER_LANCZOS4,
            cls.LINEAR  : cv2.INTER_LINEAR,
            cls.MAX     : cv2.INTER_MAX,
            cls.NEAREST : cv2.INTER_NEAREST,
        }

    @classmethod
    def pil_modes_mapping(cls) -> dict:
        """
        It maps the `InterpolationMode` enum to the corresponding PIL
        interpolation mode.
        
        Returns:
            A dictionary with the keys being the InterpolationMode enum and the
            values being the corresponding PIL interpolation mode.
        """
        return {
            cls.NEAREST : 0,
            cls.LANCZOS : 1,
            cls.BILINEAR: 2,
            cls.BICUBIC : 3,
            cls.BOX     : 4,
            cls.HAMMING : 5,
        }
    
    @classmethod
    def from_str(cls, value: str) -> InterpolationMode:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> InterpolationMode:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Enum_) -> InterpolationMode | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `InterpolationMode`, `dict`, or  `str`. "
            f"But got: {type(value)}."
        )
        return None
        
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


class MemoryUnit(Enum):
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"

    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "b" : cls.B,
            "kb": cls.KB,
            "mb": cls.MB,
            "gb": cls.GB,
            "tb": cls.TB,
            "pb": cls.PB,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.B,
            1: cls.KB,
            2: cls.MB,
            3: cls.GB,
            4: cls.TB,
            5: cls.PB,
        }
    
    @classmethod
    def byte_conversion_mapping(cls):
        """
        It returns a dictionary that maps the MemoryUnit enum to the number of
        bytes in that unit.
        
        Returns:
            A dictionary with the keys being the MemoryUnit enum and the values
            being the number of bytes in each unit.
        """
        return {
            cls.B : 1024 ** 0,
            cls.KB: 1024 ** 1,
            cls.MB: 1024 ** 2,
            cls.GB: 1024 ** 3,
            cls.TB: 1024 ** 4,
            cls.PB: 1024 ** 5,
        }
    
    @classmethod
    def from_str(cls, value: str) -> MemoryUnit:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> MemoryUnit:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(MemoryUnit.int_mapping, value)
        return MemoryUnit.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> MemoryUnit | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `MemoryUnit`, `dict`, or  `str`. "
            f"But got: {type(value)}."
        )
        return None
    
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


class ModelPhase(Enum):
    TRAINING  = "training"
    # Produce predictions, calculate losses and metrics, update weights at
    # the end of each epoch/step.
    TESTING   = "testing"
    # Produce predictions, calculate losses and metrics,
    # DO NOT update weights at the end of each epoch/step.
    INFERENCE = "inference"
    # Produce predictions ONLY.
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "training" : cls.TRAINING,
            "testing"  : cls.TESTING,
            "inference": cls.INFERENCE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.TRAINING,
            1: cls.TESTING,
            2: cls.INFERENCE,
        }

    @classmethod
    def from_str(cls, value: str) -> ModelPhase:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ModelPhase:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> ModelPhase | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `ModelPhase`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None

    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]

    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


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

    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "constant"   : cls.CONSTANT,
            "circular"   : cls.CIRCULAR,
            "reflect"    : cls.REFLECT,
            "replicate"  : cls.REPLICATE,
            "edge"       : cls.EDGE,
            "empty"      : cls.EMPTY,
            "linear_ramp": cls.LINEAR_RAMP,
            "maximum"    : cls.MAXIMUM,
            "mean"       : cls.MEAN,
            "median"     : cls.MEDIAN,
            "minimum"    : cls.MINIMUM,
            "symmetric"  : cls.SYMMETRIC,
            "wrap"       : cls.WRAP,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0 : cls.CONSTANT,
            1 : cls.CIRCULAR,
            2 : cls.REFLECT,
            3 : cls.REPLICATE,
            4 : cls.EDGE,
            5 : cls.EMPTY,
            6 : cls.LINEAR_RAMP,
            7 : cls.MAXIMUM,
            8 : cls.MEAN,
            9 : cls.MEDIAN,
            10: cls.MINIMUM,
            11: cls.SYMMETRIC,
            12: cls.WRAP,
        }

    @classmethod
    def from_str(cls, value: str) -> PaddingMode:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value)
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> PaddingMode:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> PaddingMode | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `PaddingMode`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None
        
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


class Reduction(Enum):
    NONE         = "none"
    MEAN         = "mean"
    SUM          = "sum"
    WEIGHTED_SUM = "weighted_sum"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "none"        : cls.NONE,
            "mean"        : cls.MEAN,
            "sum"         : cls.SUM,
            "weighted_sum": cls.WEIGHTED_SUM
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.NONE,
            1: cls.MEAN,
            2: cls.SUM,
            3: cls.WEIGHTED_SUM,
        }

    @classmethod
    def from_str(cls, value: str) -> Reduction:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value)
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> Reduction:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> Reduction | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, Reduction):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `Reduction`, `dict`, `str`.  "
            f"But got: {type(value)}."
        )
        return None
        
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]
    

class RGB(Enum):
    """
    Define 138 colors.
    """
    MAROON                  = (128,   0,   0)
    DARK_RED                = (139,   0,   0)
    BROWN                   = (165,  42,  42)
    FIREBRICK               = (178,  34,  34)
    CRIMSON                 = (220,  20,  60)
    RED                     = (255,   0,   0)
    TOMATO                  = (255,  99,  71)
    CORAL                   = (255, 127,  80)
    INDIAN_RED              = (205,  92,  92)
    LIGHT_CORAL             = (240, 128, 128)
    DARK_SALMON             = (233, 150, 122)
    SALMON                  = (250, 128, 114)
    LIGHT_SALMON            = (255, 160, 122)
    ORANGE_RED              = (255,  69,   0)
    DARK_ORANGE             = (255, 140,   0)
    ORANGE                  = (255, 165,   0)
    GOLD                    = (255, 215,   0)
    DARK_GOLDEN_ROD         = (184, 134,  11)
    GOLDEN_ROD              = (218, 165,  32)
    PALE_GOLDEN_ROD         = (238, 232, 170)
    DARK_KHAKI              = (189, 183, 107)
    KHAKI                   = (240, 230, 140)
    OLIVE                   = (128, 128,   0)
    YELLOW                  = (255, 255,   0)
    YELLOW_GREEN            = (154, 205,  50)
    DARK_OLIVE_GREEN        = ( 85, 107,  47)
    OLIVE_DRAB              = (107, 142,  35)
    LAWN_GREEN              = (124, 252,   0)
    CHART_REUSE             = (127, 255,   0)
    GREEN_YELLOW            = (173, 255,  47)
    DARK_GREEN              = (  0, 100,   0)
    GREEN                   = (  0, 128,   0)
    FOREST_GREEN            = ( 34, 139,  34)
    LIME                    = (  0, 255,   0)
    LIME_GREEN              = ( 50, 205,  50)
    LIGHT_GREEN             = (144, 238, 144)
    PALE_GREEN              = (152, 251, 152)
    DARK_SEA_GREEN          = (143, 188, 143)
    MEDIUM_SPRING_GREEN     = (  0, 250, 154)
    SPRING_GREEN            = (  0, 255, 127)
    SEA_GREEN               = ( 46, 139,  87)
    MEDIUM_AQUA_MARINE      = (102, 205, 170)
    MEDIUM_SEA_GREEN        = ( 60, 179, 113)
    LIGHT_SEA_GREEN         = ( 32, 178, 170)
    DARK_SLATE_GRAY         = ( 47,  79,  79)
    TEAL                    = (  0, 128, 128)
    DARK_CYAN               = (  0, 139, 139)
    AQUA                    = (  0, 255, 255)
    CYAN                    = (  0, 255, 255)
    LIGHT_CYAN              = (224, 255, 255)
    DARK_TURQUOISE          = (  0, 206, 209)
    TURQUOISE               = ( 64, 224, 208)
    MEDIUM_TURQUOISE        = ( 72, 209, 204)
    PALE_TURQUOISE          = (175, 238, 238)
    AQUA_MARINE             = (127, 255, 212)
    POWDER_BLUE             = (176, 224, 230)
    CADET_BLUE              = ( 95, 158, 160)
    STEEL_BLUE              = ( 70, 130, 180)
    CORN_FLOWER_BLUE        = (100, 149, 237)
    DEEP_SKY_BLUE           = (  0, 191, 255)
    DODGER_BLUE             = ( 30, 144, 255)
    LIGHT_BLUE              = (173, 216, 230)
    SKY_BLUE                = (135, 206, 235)
    LIGHT_SKY_BLUE          = (135, 206, 250)
    MIDNIGHT_BLUE           = ( 25,  25, 112)
    NAVY                    = (  0,   0, 128)
    DARK_BLUE               = (  0,   0, 139)
    MEDIUM_BLUE             = (  0,   0, 205)
    BLUE                    = (  0,   0, 255)
    ROYAL_BLUE              = ( 65, 105, 225)
    BLUE_VIOLET             = (138,  43, 226)
    INDIGO                  = ( 75,   0, 130)
    DARK_SLATE_BLUE         = ( 72,  61, 139)
    SLATE_BLUE              = (106,  90, 205)
    MEDIUM_SLATE_BLUE       = (123, 104, 238)
    MEDIUM_PURPLE           = (147, 112, 219)
    DARK_MAGENTA            = (139,   0, 139)
    DARK_VIOLET             = (148,   0, 211)
    DARK_ORCHID             = (153,  50, 204)
    MEDIUM_ORCHID           = (186,  85, 211)
    PURPLE                  = (128,   0, 128)
    THISTLE                 = (216, 191, 216)
    PLUM                    = (221, 160, 221)
    VIOLET                  = (238, 130, 238)
    MAGENTA                 = (255,   0, 255)
    ORCHID                  = (218, 112, 214)
    MEDIUM_VIOLET_RED       = (199,  21, 133)
    PALE_VIOLET_RED         = (219, 112, 147)
    DEEP_PINK               = (255,  20, 147)
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
    SADDLE_BROWN            = (139,  69,  19)
    SIENNA                  = (160,  82,  45)
    CHOCOLATE               = (210, 105,  30)
    PERU                    = (205, 133,  63)
    SANDY_BROWN             = (244, 164,  96)
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
    BLACK                   = (  0,   0,   0)
    DIM_GRAY                = (105, 105, 105)
    GRAY                    = (128, 128, 128)
    DARK_GRAY               = (169, 169, 169)
    SILVER                  = (192, 192, 192)
    LIGHT_GRAY              = (211, 211, 211)
    GAINSBORO               = (220, 220, 220)
    WHITE_SMOKE             = (245, 245, 245)
    WHITE                   = (255, 255, 255)
    
    @classmethod
    def random(cls):
        """
        Return a random RGB enum.
        
        Returns:
            A random choice from the list of RGB.
        """
        return random.choice(list(cls))
    
    @classmethod
    def random_value(cls):
        """
        Return a random RGB value.
        
        Returns:
            A random choice from the list of RGB.
        """
        return cls.random().value

    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.

        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]
    

class VideoFormat(Enum):
    AVI  = ".avi"
    M4V  = ".m4v"
    MKV  = ".mkv"
    MOV  = ".mov"
    MP4  = ".mp4"
    MPEG = ".mpeg"
    MPG  = ".mpg"
    WMV  = ".wmv"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "avi" : cls.AVI,
            "m4v" : cls.M4V,
            "mkv" : cls.MKV,
            "mov" : cls.MOV,
            "mp4" : cls.MP4,
            "mpeg": cls.MPEG,
            "mpg" : cls.MPG,
            "wmv" : cls.WMV,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.AVI,
            1: cls.M4V,
            2: cls.MKV,
            3: cls.MOV,
            4: cls.MP4,
            5: cls.MPEG,
            6: cls.MPG,
            7: cls.WMV,
        }
    
    @classmethod
    def from_str(cls, value: str) -> VideoFormat:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> VideoFormat:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> VideoFormat | None:
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
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `ImageFormat`, `dict`, or  `str`. "
            f"But got: {type(value)}."
        )
        return None
    
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]
    

class VisionBackend(Enum):
    CV      = "cv"
    FFMPEG  = "ffmpeg"
    LIBVIPS = "libvips"
    PIL     = "pil"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "cv"     : cls.CV,
            "ffmpeg" : cls.FFMPEG,
            "libvips": cls.LIBVIPS,
            "pil"    : cls.PIL,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.CV,
            1: cls.FFMPEG,
            2: cls.LIBVIPS,
            3: cls.PIL,
        }
    
    @classmethod
    def from_str(cls, value: str) -> VisionBackend:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping, value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> VisionBackend:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping, value)
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> VisionBackend:
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
            return cls.from_int(value)
        if isinstance(value, str):
            return cls.from_str(value)
        error_console.log(
            f"`value` must be `VisionBackend`, `dict`, or  `str`. "
            f"But got: {type(value)}."
        )
        from one.constants import VISION_BACKEND
        return VISION_BACKEND
    
    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


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


def is_collection_contain_item(input: Any, item: Any) -> bool:
    assert_collection(input)
    if item in input:
        return True
    else:
        return False
    

def is_dict(input: Any) -> bool:
    return isinstance(input, (dict, Munch))


def is_dict_contain_key(input: Any, key: Any) -> bool:
    assert_dict(input)
    if key in input:
        return True
    else:
        return False


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
    
    
def is_even_number(input: Any) -> bool:
    assert_number(input)
    return input % 2 == 0.0


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


def is_larger_than(input: Number, value: Number) -> bool:
    assert_number(input)
    assert_number(value)
    return input > value


def is_larger_or_equal_than(input: Number, value: Number) -> bool:
    assert_number(input)
    assert_number(value)
    return input >= value


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


def is_odd_number(input: Any) -> bool:
    assert_number(input)
    return input % 2 != 0.0


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


def is_smaller_than(input: Number, value: Number) -> bool:
    assert_number(input)
    assert_number(value)
    return input < value


def is_smaller_or_equal_than(input: Number, value: Number) -> bool:
    assert_number(input)
    assert_number(value)
    return input <= value


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


def is_tensor_of_shape(input: Any, shape: Ints) -> bool:
    assert_tensor(input)
    assert_list(shape)
    s = list(input.shape)
    assert_same_length(s, shape)
    return all(s1 == s2 for s1, s2 in zip(s, shape))
   

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


def is_valid_type(input: Any) -> bool:
    return isinstance(input, type)


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
        raise ValueError(f"`path` must be a bmp image file. But got: {path}.")


def assert_basename(path: Path_ | None):
    if not is_basename(path):
        raise ValueError(
            f"`path` must be a basename, i.e., name without file extension. "
            f"But got: {path}."
        )


def assert_ckpt_file(path: Path_ | None):
    if not is_ckpt_file(path):
        raise ValueError(
            f"`path` must be a checkpoint file, i.e., .ckpt "
            f"But got: {path}."
        )
    
    
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


def assert_collection_contain_item(input: Any, item: Any):
    if not is_collection_contain_item(input, item):
        raise ValueError(
            f"`input` collection must contain the item `{item}`. "
            f"But got: {item} not in {input}."
        )
    

def assert_dict(input: Any):
    if not is_dict(input):
        raise TypeError(f"`input` must be a `dict`. But got: {type(input)}.")


def assert_dict_contain_key(input: Any, key: Any):
    if not is_dict_contain_key(input, key):
        raise ValueError(
            f"`input` collection must contain the item `{key}`. "
            f"But got: {key} not in {input.keys()}."
        )


def assert_dict_of(input: dict, item_type: type):
    if not is_dict_of(input, item_type):
        raise TypeError(
            f"`input` must be a `dict` of {item_type}. "
            f"But got: {type(input)}."
        )


def assert_dir(path: Path_ | None):
    if not is_dir(path):
        raise ValueError(f"`path` must be a directory. But got: {path}.")


def assert_even_number(input: Any):
    if not is_even_number(input):
        raise ValueError(
            f"`input` must be an even number. But got: {input}."
        )


def assert_float(input: Any):
    if not is_float(input):
        raise TypeError(
            f"`input` must be a `float` number. But got: {type(input)}."
        )


def assert_image_file(path: Path_ | None):
    if not is_image_file(path):
        raise ValueError(f"`path` must be an image file. But got: {path}.")
    
    
def assert_int(input: Any):
    if not is_int(input):
        raise TypeError(
            f"`input` must be an `int` number. But got: {type(input)}."
        )


def assert_iterable(input: Any):
    if not is_iterable(input):
        raise TypeError(
            f"`inputs` must be an iterable object. But got: {type(input)}."
        )


def assert_json_file(path: Path_ | None):
    if not is_json_file(path):
        raise ValueError(f"`path` must be a json file. But got: {path}.")


def assert_larger_than(input: Number, value: Number):
    if not is_larger_than(input, value):
        raise ValueError(
            f"Expect `input` > `value`. But got: {input} > {value}."
        )


def assert_larger_or_equal_than(input: Number, value: Number):
    if not is_larger_or_equal_than(input, value):
        raise ValueError(
            f"Expect `input` >= `value`. But got: {input} > {value}."
        )


def assert_list(input: Any):
    if not is_list(input):
        raise TypeError(f"`input` must be a `list`. But got: {type(input)}.")


def assert_list_of(input: Any, item_type: type):
    if not is_list_of(input, item_type):
        raise TypeError(
            f"`input` must be a `list` of {item_type}. But got: {type(input)}"
        )


def assert_name(path: Path_ | None):
    if not is_name(path):
        raise ValueError(
            f"`path` must be a name, i.e., name with file extension. "
            f"But got: {path}."
        )


def assert_negative_number(input: Any):
    if not is_negative_number(input):
        raise ValueError(f"`input` must be negative number. But got: {input}.")


def assert_number(input: Any):
    if not is_number(input):
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
            f"Expect {start} <= `input` <= {end}. But got: {input}."
        )


def assert_numpy(input: Any):
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
            f"Expect `input.ndim` == {ndim}. But got: {input.ndim}."
        )


def assert_numpy_of_ndim_in_range(input: Any, start: int, end: int):
    if not is_numpy_of_ndim_in_range(input, start, end):
        raise ValueError(
            f"Expect {start} <= `input.ndim` <= {end}. But got: {input.ndim}."
        )


def assert_odd_number(input: Any):
    if not is_odd_number(input):
        raise ValueError(
            f"`input` must be an odd number. But got: {input}."
        )


def assert_positive_number(input: Any):
    if not is_positive_number(input):
        raise ValueError(f"`input` must be positive number. But got: {input}.")
    

def assert_same_length(input1: Sequence, input2: Sequence):
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
  
    
def assert_sequence(input: Any):
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
    
    
def assert_smaller_than(input: Number, value: Number):
    if not is_smaller_than(input, value):
        raise ValueError(
            f"Expect `input` < `value`. But got: {input} > {value}."
        )


def assert_smaller_or_equal_than(input: Number, value: Number):
    if not is_smaller_or_equal_than(input, value):
        raise ValueError(
            f"Expect `input` <= `value`. But got: {input} > {value}."
        )


def assert_stem(path: Path_ | None):
    if not is_stem(path):
        raise ValueError(
            f"`path` must be a name, i.e., name without file extension. "
            f"But got: {path}."
        )

    
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
            f"Expect {start} <= `input.ndim` <= {end}. But got: {input.ndim}."
        )


def assert_tensor_of_shape(input: Any, shape: Ints):
    if not is_tensor_of_shape(input, shape):
        raise ValueError(
            f"`input` must be of shape {shape}. But got: {list(input.shape)}."
        )


def assert_torch_saved_file(path: Path_ | None):
    if not is_torch_saved_file(path):
        raise ValueError(
            f"`path` must be a torch saved file, i.e., .pth or .ckpt. "
            f"But got: {path}."
        )
    
    
def assert_tuple(input: Any):
    if not is_tuple(input):
        raise TypeError(f"`input` must be a `tuple`. But got: {type(input)}.")
    
    
def assert_tuple_of(input: Any, item_type: type):
    if not is_tuple_of(input, item_type):
        raise TypeError(
            f"`input` must be a `tuple` of {item_type}. "
            f"But got: {type(input)}."
        )


def assert_txt_file(path: Path_ | None):
    if not is_txt_file(path):
        raise ValueError(f"`path` must be a txt file. But got: {path}.")


def assert_url(path: Path_ | None):
    if not is_url(path):
        raise ValueError(f"`path` must be an url. But got: {path}.")


def assert_url_or_file(path: Path_ | None):
    if not is_url_or_file(path):
        raise ValueError(f"`path` must be an url or a file. But got: {path}.")


def assert_valid_type(input: Any):
    if not is_valid_type(input):
        raise TypeError(f"`input` must be a valid type. But got: {input}.")


def assert_value_in_collection(input: Any, collection: Any):
    if not is_value_in_collection(input, collection):
        raise ValueError(
            f"`input` must be included in `collection`. "
            f"But got: {input} not in {collection}."
        )
    
    
def assert_video_file(path: Path_ | None):
    if not is_video_file(path):
        raise ValueError(f"`path` must be a video file. But got: {path}.")


def assert_video_stream(path: Path_ | None):
    if not is_video_stream(path):
        raise ValueError(f"`path` must be a video stream. But got: {path}.")


def assert_weights_file(path: Path_ | None):
    if not is_weights_file(path):
        raise ValueError(
            f"`path` must be a weight file, i.e., .pth file. But got: {path}."
        )


def assert_xml_file(path: Path_ | None):
    if not is_xml_file(path):
        raise ValueError(f"`path` must be a xml file. But got: {path}.")


def assert_yaml_file(path: Path_ | None):
    if not is_yaml_file(path):
        raise ValueError(f"`path` must be a yaml file. But got: {path}.")
    

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
        input = np.squeeze(input, axis=0)  # [1, *, B, H, W, C] -> [*, B, H, W, C]
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
        raise ValueError(f"Expect `n` > 0. But got: {n}.")
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
            raise ValueError(f"Expect `input.ndim` == 2.")
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
                f"Expect 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]                       # list[Tensor any dimensions] -> list[np.ndarray any dimensions]
        
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        return input                                                            # list[np.ndarray any dimensions] -> list[3D np.ndarray]
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
            raise ValueError(f"Expect 2 <= `input.ndim` <= 3.")
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
                f"Expect 3 <= `input.ndim` <= 4. But got: {input.ndim}."
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
            raise ValueError(f"Expect 2 <= `input.ndim` <= 3.")
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
            raise ValueError(f"Expect 3 <= `input.ndim` <= 5.")
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
            raise ValueError(f"Expect 2 <= `input.ndim` <= 3.")
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
            raise ValueError(f"Expect 3 <= `input.ndim` <= 5.")
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
            raise ValueError(f"Expect 2 <= `input.ndim` <= 4.")
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
            raise ValueError(f"Expect 2 <= `input.ndim` <= 4.")
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
        raise ValueError(f"Expect `n` > 0. But got: {n}.")
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
            instance.classname = name
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


def get_next_version(root_dir: str) -> int:
    """
    Get the next experiment version number.
    
    Args:
        root_dir (str): Path to the folder that contains all experiment folders.

    Returns:
         Next version number.
    """
    try:
        listdir_info = os.listdir(root_dir)
    except OSError:
        # console.log(f"Missing folder: {root_dir}")
        return 0
    
    existing_versions = []
    for listing in listdir_info:
        if isinstance(listing, str):
            d = listing
        elif isinstance(listing, dict):
            d = listing["name"]
        else:
            d = ""
        bn = os.path.basename(d)
        if bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    
    return max(existing_versions) + 1


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
    color_system    = "auto",
    log_time_format = "[%x %H:%M:%S:%f]",
    soft_wrap       = True,
    theme           = rich_console_theme,
)

error_console = Console(
    color_system    = "auto",
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
        target : Tensor  | None  = None,
        dataset: Callable | None = None,
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
        target : Tensor  | None  = None,
        dataset: Callable | None = None,
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
            from one.constants import TRANSFORMS
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
        dataset: Callable | None                        = None,
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
        dataset: Callable | None                        = None,
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
StepOutput          = Union[Tensor, dict[str, Any]]
EpochOutput         = list[StepOutput]
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
BorderType_         = Union[BorderType,        str, int]
InterpolationMode_  = Union[InterpolationMode, str, int]
ModelPhase_         = Union[ModelPhase,        str, int]
PaddingMode_        = Union[PaddingMode,       str, int]
Reduction_          = Union[Reduction,         str, int]
VisionBackend_      = Union[VisionBackend,     str, int]
# Model Building Types
Losses_             = ScalarOrCollectionT[Union[_Loss,     dict]]
Metrics_            = ScalarOrCollectionT[Union[Metric,    dict]]
Optimizers_         = ScalarOrCollectionT[Union[Optimizer, dict]]
Paddings            = Union[Ints, str]
Pretrained          = Union[bool, str, dict]
Weights             = Union[Tensor, Numbers]
