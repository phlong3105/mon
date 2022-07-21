#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Define all enums, constants, custom types, assertion, and conversion.
"""

from __future__ import annotations

import collections
import functools
import inspect
import itertools
import pathlib
import sys
import types
import typing
from collections import abc
from collections import OrderedDict
from copy import copy
from enum import Enum
from numbers import Number
from typing import Any
from typing import Collection
from typing import Iterable
from typing import Sequence
from typing import TypeVar
from typing import Union

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from munch import Munch
from ordered_enum import OrderedEnum
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric


# MARK: - Enums

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
    
    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(BBoxFormat.str_mapping, value.lower())
        return BBoxFormat.str_mapping[value]
    
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
        return BBoxFormat.int_mapping[value]

    @staticmethod
    def from_value(value: BBoxFormat_) -> BBoxFormat:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (BBoxFormat_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, BBoxFormat):
            return value
        elif isinstance(value, str):
            return BBoxFormat.from_str(value)
        elif isinstance(value, int):
            return BBoxFormat.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `BBoxFormat`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
    
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
    
    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
        return DistanceMetric.str_mapping[value]
    
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
        return DistanceMetric.int_mapping[value]

    @staticmethod
    def from_value(value: DistanceMetric_) -> DistanceMetric:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (DistanceMetric_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, DistanceMetric):
            return value
        if isinstance(value, str):
            return DistanceMetric.from_str(value)
        elif isinstance(value, int):
            return DistanceMetric.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `BBoxFormat`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
        
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
    BMP  = "bmp"
    DNG	 = "dng"
    JPG  = "jpg"
    JPEG = "jpeg"
    PNG  = "png"
    PPM  = "ppm"
    TIF  = "tif"
    TIFF = "tiff"
    
    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
        return ImageFormat.str_mapping[value]
    
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
        return ImageFormat.int_mapping[value]

    @staticmethod
    def from_value(value: ImageFormat_) -> ImageFormat:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (ImageFormat_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, ImageFormat):
            return value
        if isinstance(value, str):
            return ImageFormat.from_str(value)
        elif isinstance(value, int):
            return ImageFormat.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `ImageFormat`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
    
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

    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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

    @classmethod
    @property
    def cv_modes_mapping(cls) -> dict:
        """
        It maps the `InterpolationMode` enum to the corresponding OpenCV
        interpolation mode.
        
        Args:
            cls: The class that the decorator is being applied to.
        
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

    @classmethod
    @property
    def pil_modes_mapping(cls) -> dict:
        """
        It maps the `InterpolationMode` enum to the corresponding PIL
        interpolation mode.
        
        Args:
            cls: The class that the decorator is being applied to.
        
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
        return InterpolationMode.str_mapping[value]
    
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
        return InterpolationMode.int_mapping[value]

    @staticmethod
    def from_value(value: InterpolationMode_) -> InterpolationMode:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (InterpolationMode_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, InterpolationMode):
            return value
        if isinstance(value, str):
            return InterpolationMode.from_str(value)
        elif isinstance(value, int):
            return InterpolationMode.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `InterpolationMode`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
        
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

    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
    
    @classmethod
    @property
    def byte_conversion_mapping(cls):
        """
        It returns a dictionary that maps the MemoryUnit enum to the number of
        bytes in that unit.
        
        Args:
            cls: The class that the method is being called on.
        
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
        return MemoryUnit.str_mapping[value]
    
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
        return MemoryUnit.int_mapping[value]
    
    @staticmethod
    def from_value(value: MemoryUnit_) -> MemoryUnit:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (MemoryUnit_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, MemoryUnit):
            return value
        if isinstance(value, str):
            return MemoryUnit.from_str(value)
        elif isinstance(value, int):
            return MemoryUnit.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `MemoryUnit`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
    
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


class ModelState(Enum):
    TRAINING  = "training"   # Produce predictions, calculate losses and metrics,
                             # update weights at the end of each epoch/step.
    TESTING   = "testing"    # Produce predictions, calculate losses and metrics,
                             # DO NOT update weights at the end of each epoch/step.
    INFERENCE = "inference"  # Produce predictions ONLY.
    
    @classmethod
    @property
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "training" : ModelState.TRAINING,
            "testing"  : ModelState.TESTING,
            "inference": ModelState.INFERENCE,
        }

    @classmethod
    @property
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: ModelState.TRAINING,
            1: ModelState.TESTING,
            2: ModelState.INFERENCE,
        }

    @staticmethod
    def from_str(value: str) -> ModelState:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(ModelState.str_mapping, value.lower())
        return ModelState.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> ModelState:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(ModelState.int_mapping, value)
        return ModelState.int_mapping[value]

    @staticmethod
    def from_value(value: ModelState_) -> ModelState:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (ModelState_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, ModelState):
            return value
        if isinstance(value, str):
            return ModelState.from_str(value)
        elif isinstance(value, int):
            return ModelState.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `ModelState`, `str`, or `int`. "
                f"But got: {type(value)}."
            )

    @staticmethod
    def keys() -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in ModelState]

    @staticmethod
    def values() -> list[str]:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in ModelState]
    

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
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
        return PaddingMode.str_mapping[value]
    
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
        return PaddingMode.int_mapping[value]

    @staticmethod
    def from_value(value: PaddingMode_) -> PaddingMode:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (PaddingMode_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, PaddingMode):
            return value
        if isinstance(value, str):
            return PaddingMode.from_str(value)
        elif isinstance(value, int):
            return PaddingMode.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `PaddingMode`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
        
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
    """Define 138 colors."""
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
    AVI  = "avi"
    M4V  = "m4v"
    MKV  = "mkv"
    MOV  = "mov"
    MP4  = "mp4"
    MPEG = "mpeg"
    MPG  = "mpg"
    WMV  = "wmv"
    
    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
        return VideoFormat.str_mapping[value]
    
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
        return VideoFormat.int_mapping[value]

    @staticmethod
    def from_value(value: VideoFormat_) -> VideoFormat:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (VideoFormat_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, VideoFormat):
            return value
        if isinstance(value, str):
            return VideoFormat.from_str(value)
        elif isinstance(value, int):
            return VideoFormat.from_int(value)
        else:
            raise TypeError(
                f"`value` must be `VideoFormat`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
    
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

    @classmethod
    @property
    def str_mapping(cls) -> dict:
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

    @classmethod
    @property
    def int_mapping(cls) -> dict:
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
        return VisionBackend.str_mapping[value]
    
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
        return VisionBackend.int_mapping[value]

    @staticmethod
    def from_value(value: VisionBackend_) -> VisionBackend:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (VisionBackend_): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, VisionBackend):
            return value
        elif isinstance(value, int):
            return VisionBackend.from_int(value)
        elif isinstance(value, str):
            return VisionBackend.from_str(value)
        else:
            raise TypeError(
                f"`value` must be `VisionBackend`, `str`, or `int`. "
                f"But got: {type(value)}."
            )
    
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


BBoxFormat_         = Union[BBoxFormat,        str, int]
CFA_                = Union[CFA,               str, int]
DistanceMetric_     = Union[DistanceMetric,    str, int]
ImageFormat_        = Union[ImageFormat,       str, int]
InterpolationMode_  = Union[InterpolationMode, str, int]
MemoryUnit_         = Union[MemoryUnit,        str, int]
ModelState_         = Union[ModelState,        str, int]
PaddingMode_        = Union[PaddingMode,       str, int]
VideoFormat_        = Union[VideoFormat,       str, int]
VisionBackend_      = Union[VisionBackend,     str, int]


# MARK: - Typing
# NOTE: Template
T                   = TypeVar("T")
ScalarOrSequenceT   = Union[T, Sequence[T]]
ScalarOrCollectionT = Union[T, Collection[T]]
# NOTE: Built-in Types
Callable            = Union[typing.Callable, types.FunctionType, functools.partial]
Dict                = Union[dict, Munch]
Ints                = ScalarOrSequenceT[int]
Floats              = ScalarOrSequenceT[float]
Numbers             = ScalarOrSequenceT[Number]
Strs                = ScalarOrSequenceT[str]
# NOTE: Custom Types
Color               = Sequence[int]
Devices             = Union[Ints, Strs]
Path                = Union[str, pathlib.Path, typing.TextIO]
Paths               = ScalarOrSequenceT[Path]
Tensors             = ScalarOrCollectionT[Tensor]
# NOTE: Data Types
EvalDataLoaders     = ScalarOrSequenceT[DataLoader]
TrainDataLoaders    = Union[DataLoader,
                            Sequence[DataLoader],
                            Sequence[Sequence[DataLoader]],
                            Sequence[dict[str, DataLoader]],
                            dict[str, DataLoader],
                            dict[str, dict[str, DataLoader]],
                            dict[str, Sequence[DataLoader]]]
# NOTE: Model Building Types
Losses_             = ScalarOrCollectionT[Union[_Loss,     Dict]]
Metrics_            = ScalarOrCollectionT[Union[Metric,    Dict]]
Optimizers_         = ScalarOrCollectionT[Union[Optimizer, Dict]]
Paddings            = Union[Ints, str]
Pretrained          = Union[bool, str, Dict]
Weights             = Union[Tensor, Numbers]


# MARK: - Assertion

def is_class(input: Any) -> bool:
    if inspect.isclass(input):
        return True
    else:
        raise TypeError(
            f"`input` must be a class type. But got: {type(input)}."
        )


def is_collection(input: Any) -> bool:
    if isinstance(input, Collection):
        return True
    else:
        raise TypeError(
            f"`input` must be a `Collection`. But got: {type(input)}."
        )
        

def is_dict(input: Any) -> bool:
    if isinstance(input, (dict, Munch)):
        return True
    else:
        raise TypeError(f"`input` must be a `dict`. But got: {type(input)}.")


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


def is_float(input: Any) -> bool:
    if isinstance(input, float):
        return True
    else:
        raise TypeError(f"`input` must be a `float`. But got: {type(input)}.")


def is_int(input: Any) -> bool:
    if isinstance(input, int):
        return True
    else:
        raise TypeError(f"`input` must be an `int`. But got: {type(input)}.")
    
    
def is_iterable(input: Any) -> bool:
    if isinstance(input, Iterable):
        return True
    else:
        raise TypeError(
            f"`inputs` must be an iterable object. But got: {type(input)}."
        )
    

def is_list(input: Any) -> bool:
    if isinstance(input, list):
        return True
    else:
        raise TypeError(f"`input` must be a `list`. But got: {type(input)}.")


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


def is_negative_number(input: Any) -> bool:
    assert_number(input)
    if input < 0.0:
        return True
    else:
        raise ValueError(f"`input` must be negative. But got: {input}.")
    
    
def is_number(input: Any) -> bool:
    if isinstance(input, Number):
        return True
    else:
        raise TypeError(f"`input` must be a `Number`. But got: {type(input)}.")


def is_number_divisible_to(input: Any, k: int) -> bool:
    assert_number(input)
    assert_number(k)
    if input % k == 0:
        return True
    else:
        raise ValueError(
            f"`input` must be divisible by `{k}`. "
            f"But got: {input} % {k} != 0."
        )


def is_number_in_range(input: Any, start: Number, end: Number) -> bool:
    assert_number(input)
    assert_number(start)
    assert_number(end)
    if start <= input <= end:
        return True
    else:
        raise ValueError(
            f"Require {start} <= `input` <= {end}. But got: {input}."
        )


def is_numpy(input: Any) -> bool:
    if isinstance(input, np.ndarray):
        return True
    else:
        raise TypeError(
            f"`input` must be a `np.ndarray`. But got: {type(input)}."
        )


def is_numpy_of_atleast_ndim(input: Any, ndim: int) -> bool:
    assert_numpy(input)
    assert_int(ndim)
    if input.ndim >= ndim:
        return True
    else:
        raise TypeError(
            f"`input` must be a `np.ndarray` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )
    

def is_numpy_of_channels(input: Any, channels: Ints) -> bool:
    from one.vision.transformation import get_num_channels
    assert_numpy_of_atleast_ndim(input, 3)
    
    channels = to_list(channels)
    assert_list(channels)
    
    c = get_num_channels(input)
    if c in channels:
        return True
    else:
        raise TypeError(
            f"`input` must be a `np.ndarray` of channels `{channels}`. "
            f"But got: {c}."
        )


def is_numpy_of_ndim(input: Any, ndim: int) -> bool:
    assert_numpy(input)
    assert_int(ndim)
    if input.ndim == ndim:
        return True
    else:
        raise TypeError(
            f"`input` must be a `np.ndarray` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )
    

def is_numpy_of_ndim_in_range(input: Any, start: int, end: int) -> bool:
    assert_numpy(input)
    assert_int(start)
    assert_int(end)
    if start <= input.ndim <= end:
        return True
    else:
        raise ValueError(
            f"Require {start} <= `input.ndim` <= {end}. But got: {input.ndim}."
        )


def is_positive_number(input: Any) -> bool:
    assert_number(input)
    if input >= 0.0:
        return True
    else:
        raise ValueError(f"`input` must be positive. But got: {input}.")
    

def is_same_length(input1: Sequence, input2: Sequence) -> bool:
    assert_sequence(input1)
    assert_sequence(input2)
    if len(input1) == len(input2):
        return True
    else:
        raise ValueError(
            f"`input1` and `input2` must have the same length. "
            f"But got: {len(input1)} != {len(input2)}."
        )


def is_same_shape(
    input1: Tensor | np.ndarray, input2: Tensor | np.ndarray
) -> bool:
    if input1.shape == input2.shape:
        return True
    else:
        raise ValueError(
            f"`input1` and `input2` must have the same shape. "
            f"But got: {input1.shape} != {input2.shape}."
        )


def is_sequence(input: Any) -> bool:
    if isinstance(input, Sequence):
        return True
    else:
        raise TypeError(
            f"`input` must be a `Sequence`. But got: {type(input)}."
        )


def is_sequence_of(
    input    : Sequence,
    item_type: type,
    seq_type : Union[type, None] = None
) -> bool:
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


def is_sequence_of_length(input: Any, len: int) -> bool:
    assert_sequence(input)
    assert_int(len)
    if len(input) == len:
        return True
    else:
        raise TypeError(
            f"`input` must be a sequence of length `{len}`. "
            f"But got: {len(input)}."
        )


def is_str(input: Any) -> bool:
    if isinstance(input, str):
        return True
    else:
        raise TypeError(f"`input` must be a `str`. But got: {type(input)}.")


def is_tensor(input: Any) -> bool:
    if isinstance(input, Tensor):
        return True
    else:
        raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def is_tensor_of_atleast_ndim(input: Any, ndim: int) -> bool:
    assert_tensor(input)
    assert_int(ndim)
    if input.ndim >= ndim:
        return True
    else:
        raise TypeError(
            f"`input` must be a `Tensor` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )
    

def is_tensor_of_channels(input: Any, channels: Ints) -> bool:
    from one.vision.transformation import get_num_channels
    assert_tensor_of_atleast_ndim(input, 3)
    
    channels = to_list(channels)
    assert_list(channels)
    
    c = get_num_channels(input)
    if c in channels:
        return True
    else:
        raise TypeError(
            f"`input` must be a `Tensor` of channels `{channels}`. "
            f"But got: {c}."
        )


def is_tensor_of_ndim(input: Any, ndim: int) -> bool:
    assert_tensor(input)
    assert_int(ndim)
    if input.ndim == ndim:
        return True
    else:
        raise TypeError(
            f"`input` must be a `Tensor` of ndim `{ndim}`. "
            f"But got: {input.ndim}."
        )
    

def is_tensor_of_ndim_in_range(input: Any, start: int, end: int) -> bool:
    assert_tensor(input)
    assert_int(start)
    assert_int(end)
    if start <= input.ndim <= end:
        return True
    else:
        raise ValueError(
            f"Require {start} <= `input.ndim` <= {end}. But got: {input.ndim}."
        )
    

def is_tuple(input: Any) -> bool:
    if isinstance(input, tuple):
        return True
    else:
        raise TypeError(f"`input` must be a `tuple`. But got: {type(input)}.")


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
    if isinstance(input, type):
        return True
    else:
        raise TypeError(f"`input` must be a valid type. But got: {input}.")


def is_value_in_collection(input: Any, collection: Any) -> bool:
    assert_collection(collection)
    if input in collection:
        return True
    else:
        raise ValueError(
            f"`input` must be included in `collection`. "
            f"But got: {input} not in {collection}."
        )


assert_collection              = is_collection
assert_class                   = is_class
assert_dict                    = is_dict
assert_dict_contain_key        = is_dict_contain_key
assert_dict_of                 = is_dict_of
assert_float                   = is_float
assert_int                     = is_int
assert_iterable                = is_iterable
assert_list                    = is_list
assert_list_of                 = is_list_of
assert_negative_number         = is_negative_number
assert_number                  = is_number
assert_number_divisible_to     = is_number_divisible_to
assert_number_in_range         = is_number_in_range
assert_numpy                   = is_numpy
assert_numpy_of_atleast_ndim   = is_numpy_of_atleast_ndim
assert_numpy_of_channels       = is_numpy_of_channels
assert_numpy_of_ndim           = is_numpy_of_ndim
assert_numpy_of_ndim_in_range  = is_numpy_of_ndim_in_range
assert_positive_number         = is_positive_number
assert_same_length             = is_same_length
assert_same_shape              = is_same_shape
assert_sequence                = is_sequence
assert_sequence_of             = is_sequence_of
assert_sequence_of_length      = is_sequence_of_length
assert_str                     = is_str
assert_tensor                  = is_tensor
assert_tensor_of_atleast_ndim  = is_tensor_of_atleast_ndim
assert_tensor_of_channels      = is_tensor_of_channels
assert_tensor_of_ndim          = is_tensor_of_ndim
assert_tensor_of_ndim_in_range = is_tensor_of_ndim_in_range
assert_tuple                   = is_tuple
assert_tuple_of                = is_tuple_of
assert_valid_type              = is_valid_type
assert_value_in_collection     = is_value_in_collection


# MARK: - Parsing

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
    if input.ndim == 2:  # [H, W] -> [1, H, W]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4 and input.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
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
    if input.ndim == 2:  # [H, W] -> [1, H, W, 1]
        input = np.expand_dims(input, axis=0)
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [C, H, W] -> [1, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4:  # [B, H, W, C]
        pass
    elif input.ndim == 5 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)
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
        input = input.squeeze(dim=0)
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
    if input.ndim == 2:  # [H, W] -> [1, 1, H, W, 1]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 3:  # [C, H, W] -> [1, 1, H, W, 1]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 4:  # [B, C, H, W] -> [1, B, H, W, C]
        input = np.expand_dims(input, axis=0)
    elif input.ndim == 5:  # [*, B, H, W, C]
        pass
    elif input.ndim == 6 and input.shape[0] == 1:
        input = np.squeeze(input, axis=0)
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
        input = input.squeeze(dim=0)
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


def slice_list(l: list, lens: Ints) -> list[list]:
    """
    Takes a list and a list of integers, and returns a list of lists, where
    each sublist is of length specified by the corresponding integer in the
    list of integers.
    
    Args:
        l (list): list.
        lens (Ints): The number of elements in each sublist.
    
    Returns:
        A list of lists.
    """
    if isinstance(lens, int):
        assert_number_divisible_to(len(l), lens)
        lens = [lens] * int(len(l) / lens)
    
    assert_list(lens)
    assert_same_length(lens, l)
    
    out_list = []
    idx      = 0
    for i in range(len(lens)):
        out_list.append(l[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def to_3d_array_list(input: Any) -> list[np.ndarray]:
    """
    Convert input to a list of 3D arrays.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        List of 3D arrays of shape [C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [input]
        elif input.ndim == 4:
            input = list(input)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 3 for i in input):
            return input
        else:
            raise ValueError(f"Require all `input.ndim` == 3.")
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_3d_tensor_list(input: Any) -> list[Tensor]:
    """
    Convert input to a list of 3D tensors.
   
    Args:
        input (Any): Input of arbitrary type.
            
    Returns:
        List of 3D tensors of shape [C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        if input.ndim == 3:
            input = [input.unsqueeze(dim=0)]
        elif input.ndim == 4:
            input = list(input)
        else:
            raise ValueError(
                f"Require 3 <= `input.ndim` <= 4. But got: {input.ndim}."
            )
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 3 for i in input):
            return input
        else:
            raise ValueError(f"Require all `input.ndim` == 3.")
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_4d_array(input) -> np.ndarray:
    """
    Convert input to a 4D array.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        A 4D array of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 2 for i in input):
            input = [np.expand_dims(i, axis=0) for i in input]
        elif all(i.ndim == 3 for i in input):
            input = np.stack(input)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        return _to_4d_array(input)
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
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]
        elif all(i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 3.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        return _to_4d_tensor(input)
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")
    

def to_4d_array_list(input) -> list[np.ndarray]:
    """
    Convert input to a list of 4D arrays.
   
    Args:
        input (Any): Input of arbitrary type.
        
    Returns:
        List of 4D arrays of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        if input.ndim == 3:
            input = [np.expand_dims(input, axis=0)]
        elif input.ndim == 4:
            input = [input]
        elif input.ndim == 5:
            input = list(input)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 5.")
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 3 for i in input):
            return [np.stack(input, axis=0)]
        elif all(i.ndim == 4 for i in input):
            return input
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4.")
    raise TypeError(f"`input` must be a `np.ndarray`. But got: {type(input)}.")


def to_4d_tensor_list(input) -> list[Tensor]:
    """
    Convert input to a list of 4D tensors.
   
    Args:
        input (Any): Input of arbitrary type.
            
    Returns:
        List of 4D tensors of shape [B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        if input.ndim == 3:
            input = [input.unsqueeze(dim=0)]
        elif input.ndim == 4:
            input = [input]
        elif input.ndim == 5:
            input = list(input)
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 5.")
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 3 for i in input):
            return [torch.stack(input, dim=0)]
        elif all(i.ndim == 4 for i in input):
            return input
        else:
            raise ValueError(f"Require 3 <= `input.ndim` <= 4.")
    raise TypeError(f"`input` must be a `Tensor`. But got: {type(input)}.")


def to_5d_array(input) -> np.ndarray:
    """
    Convert input to a 5D array.
    
    Args:
        input (Any): Input of arbitrary type.
    
    Returns:
        A 5D array of shape [*, B, C, H, W].
    """
    if isinstance(input, dict):
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, Tensor):
        input = [i.detach().cpu().numpy() for i in input]
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        if all(i.ndim == 2 for i in input):
            input = [np.expand_dims(i, axis=0) for i in input]
        elif all(3 <= i.ndim <= 4 for i in input):
            input = np.stack(input)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 4.")
    if isinstance(input, Tensor):
        input = input.detach().cpu().numpy()
    if isinstance(input, np.ndarray):
        return _to_5d_array(input)
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
        input = list(input.values())
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list) and is_list_of(input, np.ndarray):
        input = [torch.from_numpy(i) for i in input]
    if isinstance(input, list) and is_list_of(input, Tensor):
        if all(i.ndim == 2 for i in input):
            input = [i.unsqueeze(dim=0) for i in input]
        if all(3 <= i.ndim <= 4 for i in input):
            input = torch.stack(input, dim=0)
        else:
            raise ValueError(f"Require 2 <= `input.ndim` <= 4.")
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(input, Tensor):
        return _to_5d_tensor(input)
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
    

def to_ntuple(n: int) -> tuple:
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
            size = size[0:2]
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
def unique(l: list) -> list:
    """
    Return a list with only unique items.

    Args:
        l (list): list.
    
    Returns:
        A list containing only unique items.
    """
    return list(set(l))


@dispatch(tuple)
def unique(t: tuple) -> tuple:
    """
    Return a tuple containing the unique elements of the given tuple.
    
    Args:
        t (tuple): tuple.
    
    Returns:
        A tuple of unique elements from the input tuple.
    """
    return tuple(set(t))


to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)
to_5tuple = to_ntuple(5)
to_6tuple = to_ntuple(6)


# MARK: - Constants

DEFAULT_CROP_PCT        = 0.875
IMAGENET_DEFAULT_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD    = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD  = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN       = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD        = tuple([1 / (0.0167 * 255)] * 3)

PI                      = torch.tensor(3.14159265358979323846)
VISION_BACKEND          = VisionBackend.PIL


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
