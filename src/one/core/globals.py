#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A global namespace to store all constants, enums, and typing.
"""

from __future__ import annotations

import functools
import types
from enum import Enum
from typing import Any
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import cv2
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric

from one.core.factory import Factory
from one.core.factory import OptimizerFactory
from one.core.factory import SchedulerFactory

# MARK: - Typing
# NOTE: Base
# Template for arguments which can be supplied as a tuple, or which can be a
# scalar which PyTorch will internally broadcast to a tuple. Comes in several
# variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d
# operations.
T                      = TypeVar("T")
Callable               = Union[str, type, object, types.FunctionType, functools.partial]
ScalarOrTuple1T        = Union[T, tuple[T]]
ScalarOrTuple2T        = Union[T, tuple[T, T]]
ScalarOrTuple3T        = Union[T, tuple[T, T, T]]
ScalarOrTuple4T        = Union[T, tuple[T, T, T, T]]
ScalarOrTuple5T        = Union[T, tuple[T, T, T, T, T]]
ScalarOrTuple6T        = Union[T, tuple[T, T, T, T, T, T]]
ScalarOrTupleAnyT      = Union[T, tuple[T, ...]]
ScalarListOrTuple1T    = Union[T, list[T], tuple[T]]
ScalarListOrTuple2T    = Union[T, list[T], tuple[T, T]]
ScalarListOrTuple3T    = Union[T, list[T], tuple[T, T, T]]
ScalarListOrTuple4T    = Union[T, list[T], tuple[T, T, T, T]]
ScalarListOrTuple5T    = Union[T, list[T], tuple[T, T, T, T, T]]
ScalarListOrTuple6T    = Union[T, list[T], tuple[T, T, T, T, T, T]]
ScalarListOrTupleAnyT  = Union[T, list[T], tuple[T, ...]]
ScalarOrCollectionAnyT = Union[T, list[T], tuple[T, ...], dict[Any, T]]
ListOrTupleAnyT        = Union[   list[T], tuple[T, ...]]
ListOrTuple1T          = Union[   list[T], tuple[T]]
ListOrTuple2T          = Union[   list[T], tuple[T, T]]
ListOrTuple3T          = Union[   list[T], tuple[T, T, T]]
ListOrTuple4T          = Union[   list[T], tuple[T, T, T, T]]
ListOrTuple5T          = Union[   list[T], tuple[T, T, T, T, T]]
ListOrTuple6T          = Union[   list[T], tuple[T, T, T, T, T, T]]
ListOrTuple3or4T       = Union[   list[T], tuple[T, T, T], tuple[T, T, T, T]]
# NOTE: Custom
Array1T                = ScalarListOrTuple1T[np.ndarray]
Array2T                = ScalarListOrTuple2T[np.ndarray]
Array3T                = ScalarListOrTuple3T[np.ndarray]
Array4T                = ScalarListOrTuple4T[np.ndarray]
Array5T                = ScalarListOrTuple5T[np.ndarray]
Array6T                = ScalarListOrTuple6T[np.ndarray]
ArrayAnyT              = ScalarListOrTupleAnyT[np.ndarray]
ArrayList              = list[np.ndarray]
Arrays                 = ScalarOrCollectionAnyT[np.ndarray]
Augment_               = Union[dict, Callable]
Color                  = ListOrTuple3or4T[int]
Devices                = Union[ScalarListOrTupleAnyT[int],
                               ScalarListOrTupleAnyT[str]]
EvalDataLoaders        = Union[DataLoader, Sequence[DataLoader]]
Int1T                  = ScalarListOrTuple1T[int]
Int2T                  = ScalarListOrTuple2T[int]
Int3T                  = ScalarListOrTuple3T[int]
Int4T                  = ScalarListOrTuple4T[int]
Int5T                  = ScalarListOrTuple5T[int]
Int6T                  = ScalarListOrTuple6T[int]
IntAnyT                = ScalarListOrTupleAnyT[int]
Int2Or3T               = Union[Int2T, Int3T]
Float1T                = ScalarListOrTuple1T[float]
Float2T                = ScalarListOrTuple2T[float]
Float3T                = ScalarListOrTuple3T[float]
Float4T                = ScalarListOrTuple4T[float]
Float5T                = ScalarListOrTuple5T[float]
Float6T                = ScalarListOrTuple6T[float]
FloatAnyT              = ScalarListOrTupleAnyT[float]
Indexes                = ScalarListOrTupleAnyT[int]
Losses_                = Union[_Loss,  dict, list[Union[_Loss,  dict]]]
Metrics_               = Union[Metric, dict, list[Union[Metric, dict]]]
Number                 = Union[int, float]
Optimizers_            = Union[Optimizer, dict, list[Union[Optimizer, dict]]]
Padding1T              = Union[str, ScalarListOrTuple1T[int]]
Padding2T              = Union[str, ScalarListOrTuple2T[int]]
Padding3T              = Union[str, ScalarListOrTuple3T[int]]
Padding4T              = Union[str, ScalarListOrTuple4T[int]]
Padding5T              = Union[str, ScalarListOrTuple5T[int]]
Padding6T              = Union[str, ScalarListOrTuple6T[int]]
PaddingAnyT            = Union[str, ScalarListOrTupleAnyT[int]]
Pretrained             = Union[bool, str, dict]
Tasks                  = ScalarListOrTupleAnyT[str]
Tensor1T               = ScalarListOrTuple1T[Tensor]
Tensor2T               = ScalarListOrTuple2T[Tensor]
Tensor3T               = ScalarListOrTuple3T[Tensor]
Tensor4T               = ScalarListOrTuple4T[Tensor]
Tensor5T               = ScalarListOrTuple5T[Tensor]
Tensor6T               = ScalarListOrTuple6T[Tensor]
TensorAnyT             = ScalarListOrTupleAnyT[Tensor]
TensorList             = list[Tensor]
Tensors                = ScalarOrCollectionAnyT[Tensor]
TensorOrArray          = Union[Tensor, np.ndarray]
TensorsOrArrays        = Union[Tensors, Arrays]
TrainDataLoaders       = Union[
    DataLoader,
    Sequence[DataLoader],
    Sequence[Sequence[DataLoader]],
    Sequence[dict[str, DataLoader]],
    dict[str, DataLoader],
    dict[str, dict[str, DataLoader]],
    dict[str, Sequence[DataLoader]],
]
Transform_             = Union[dict, Callable]
Transforms_            = Union[str, nn.Sequential, Transform_, list[Transform_]]
Weights                = Union[Tensor,
                               ListOrTupleAnyT[float],
                               ListOrTupleAnyT[int]]

ForwardOutput          = tuple[TensorsOrArrays, Optional[TensorsOrArrays]]
StepOutput             = Union[TensorsOrArrays, dict]
EpochOutput            = list[StepOutput]
EvalOutput             = list[dict]
PredictOutput          = Union[list[Any], list[list[Any]]]


# MARK: - Enums

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
        value = value.lower()
        if value not in DistanceMetric.str_mapping:
            raise ValueError(f"`value` must be one of: {DistanceMetric.str_mapping.keys()}. "
                             f"But got: {value}.")
        return DistanceMetric.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> DistanceMetric:
        if value not in DistanceMetric.int_mapping:
            raise ValueError(f"`value` must be one of: {DistanceMetric.int_mapping.keys()}. "
                             f"But got: {value}.")
        return DistanceMetric.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> DistanceMetric:
        if isinstance(value, int):
            return DistanceMetric.from_int(value)
        else:
            return DistanceMetric.from_str(value)
        
    @staticmethod
    def keys():
        return [e for e in DistanceMetric]
    
    @staticmethod
    def values() -> list[str]:
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
        value = value.lower()
        if value not in ImageFormat.str_mapping:
            raise ValueError(f"`value` must be one of: {ImageFormat.str_mapping.keys()}. "
                             f"But got: {value}.")
        return ImageFormat.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> ImageFormat:
        if value not in ImageFormat.int_mapping:
            raise ValueError(f"`value` must be one of: {ImageFormat.int_mapping.keys()}. "
                             f"But got: {value}.")
        return ImageFormat.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> ImageFormat:
        if isinstance(value, int):
            return ImageFormat.from_int(value)
        else:
            return ImageFormat.from_str(value)
    
    @staticmethod
    def keys():
        return [e for e in ImageFormat]
    
    @staticmethod
    def values() -> list[str]:
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
        value = value.lower()
        if value not in InterpolationMode.str_mapping:
            raise ValueError(f"`value` must be one of: {InterpolationMode.str_mapping.keys()}. "
                             f"But got: {value}.")
        return InterpolationMode.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> InterpolationMode:
        if value not in InterpolationMode.int_mapping:
            raise ValueError(f"`value` must be one of: {InterpolationMode.int_mapping.keys()}. "
                             f"But got: {value}.")
        return InterpolationMode.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> InterpolationMode:
        if isinstance(value, int):
            return InterpolationMode.from_int(value)
        else:
            return InterpolationMode.from_str(value)
        
    @staticmethod
    def keys():
        return [e for e in InterpolationMode]
    
    @staticmethod
    def values() -> list[str]:
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
        value = value.lower()
        if value not in MemoryUnit.str_mapping:
            raise ValueError(f"`value` must be one of: {MemoryUnit.str_mapping.keys()}. "
                             f"But got: {value}.")
        return MemoryUnit.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> MemoryUnit:
        if value not in MemoryUnit.int_mapping:
            raise ValueError(f"`value` must be one of: {MemoryUnit.int_mapping.keys()}. "
                             f"But got: {value}.")
        return MemoryUnit.int_mapping[value]
    
    @staticmethod
    def from_value(value: Union[str, int]) -> MemoryUnit:
        if isinstance(value, int):
            return MemoryUnit.from_int(value)
        else:
            return MemoryUnit.from_str(value)
    
    @staticmethod
    def keys():
        return [e for e in MemoryUnit]
    
    @staticmethod
    def values() -> list[str]:
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
        return {
            "training" : ModelState.TRAINING,
            "testing"  : ModelState.TESTING,
            "inference": ModelState.INFERENCE,
        }

    @classmethod
    @property
    def int_mapping(cls) -> dict:
        return {
            0: ModelState.TRAINING,
            1: ModelState.TESTING,
            2: ModelState.INFERENCE,
        }

    @staticmethod
    def from_str(value: str) -> ModelState:
        if value not in ModelState.str_mapping:
            raise ValueError(f"`value` must be one of: {ModelState.str_mapping.keys()}. "
                             f"But got: {value}.")
        return ModelState.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> ModelState:
        if value not in ModelState.int_mapping:
            raise ValueError(f"`value` must be one of: {ModelState.int_mapping.keys()}. "
                             f"But got: {value}.")
        return ModelState.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> ModelState:
        if isinstance(value, int):
            return ModelState.from_int(value)
        else:
            return ModelState.from_str(value)
    
    @staticmethod
    def values() -> list[str]:
        """Return the list of all values."""
        return [e.value for e in ModelState]
    
    @staticmethod
    def keys():
        """Return the list of all enum keys."""
        return [e for e in ModelState]
    

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
        if value not in PaddingMode.str_mapping:
            raise ValueError(f"`value` must be one of: {PaddingMode.str_mapping.keys()}. "
                             f"But got: {value}.")
        return PaddingMode.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> PaddingMode:
        if value not in PaddingMode.int_mapping:
            raise ValueError(f"`value` must be one of: {PaddingMode.int_mapping.keys()}. "
                             f"But got: {value}.")
        return PaddingMode.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> PaddingMode:
        if isinstance(value, int):
            return PaddingMode.from_int(value)
        else:
            return PaddingMode.from_str(value)
        
    @staticmethod
    def keys():
        return [e for e in PaddingMode]
    
    @staticmethod
    def values() -> list[str]:
        return [e.value for e in PaddingMode]


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
        value = value.lower()
        if value not in VideoFormat.str_mapping:
            raise ValueError(f"`value` must be one of: {VideoFormat.str_mapping.keys()}. "
                             f"But got: {value}.")
        return VideoFormat.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> VideoFormat:
        if value not in VideoFormat.int_mapping:
            raise ValueError(f"`value` must be one of: {VideoFormat.int_mapping.keys()}. "
                             f"But got: {value}.")
        return VideoFormat.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> VideoFormat:
        if isinstance(value, int):
            return VideoFormat.from_int(value)
        else:
            return VideoFormat.from_str(value)
    
    @staticmethod
    def keys():
        return [e for e in VideoFormat]
    
    @staticmethod
    def values() -> list[str]:
        return [e.value for e in VideoFormat]
    

class VisionBackend(Enum):
    CV      = "cv"
    FFMPEG  = "ffmpeg"
    LIBVIPS = "libvips"
    PIL     = "pil"

    @classmethod
    @property
    def str_mapping(cls) -> dict:
        return {
            "cv"     : VisionBackend.CV,
            "ffmpeg" : VisionBackend.FFMPEG,
            "libvips": VisionBackend.LIBVIPS,
            "pil"    : VisionBackend.PIL,
        }

    @classmethod
    @property
    def int_mapping(cls) -> dict:
        return {
            0: VisionBackend.CV,
            1: VisionBackend.FFMPEG,
            2: VisionBackend.LIBVIPS,
            3: VisionBackend.PIL,
        }
    
    @staticmethod
    def from_str(value: str) -> VisionBackend:
        if value not in VisionBackend.str_mapping:
            raise ValueError(f"`value` must be one of: {VisionBackend.str_mapping.keys()}. "
                             f"But got: {value}.")
        return VisionBackend.str_mapping[value]
    
    @staticmethod
    def from_int(value: int) -> VisionBackend:
        if value not in VisionBackend.int_mapping:
            raise ValueError(f"`value` must be one of: {VisionBackend.int_mapping.keys()}. "
                             f"But got: {value}.")
        return VisionBackend.int_mapping[value]

    @staticmethod
    def from_value(value: Union[str, int]) -> VisionBackend:
        if isinstance(value, int):
            return VisionBackend.from_int(value)
        else:
            return VisionBackend.from_str(value)
    
    @staticmethod
    def keys():
        return [e for e in VisionBackend]

    @staticmethod
    def values() -> list[str]:
        return [e.value for e in VisionBackend]


# MARK: - Builders
# NOTE: NN Layers
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
# NOTE: Models
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
# NOTE: Misc
AUGMENTS             = Factory(name="augments")
DATAMODULES          = Factory(name="datamodules")
DATASETS             = Factory(name="datasets")
DISTANCES            = Factory(name="distance_functions")
DISTANCE_FUNCS       = Factory(name="distance_functions")
FILE_HANDLERS        = Factory(name="file_handler")
LABEL_HANDLERS       = Factory(name="label_handlers")
MOTIONS              = Factory(name="motions")
TRANSFORMS           = Factory(name="transforms")


# MARK: - Constants

DEFAULT_CROP_PCT        = 0.875
IMAGENET_DEFAULT_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD    = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD  = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN       = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD        = tuple([1 / (.0167 * 255)] * 3)

PI             = torch.tensor(3.14159265358979323846)
VISION_BACKEND = VisionBackend.PIL
