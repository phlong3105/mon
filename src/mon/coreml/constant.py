#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`mon.coreml`
package.
"""

from __future__ import annotations

__all__ = [
    "ACCELERATOR", "CALLBACK", "DATAMODULE", "DATASET", "LAYER", "LOGGER",
    "LOSS", "LR_SCHEDULER", "MODEL", "METRIC", "ModelPhase", "OPTIMIZER",
    "Reduction", "STRATEGY", "TRANSFORM",
    # Extend :mod:`mon.core.constant`
    "CONTENT_ROOT_DIR", "DATA_DIR", "DOCS_DIR", "FILE_HANDLER", "ImageFormat",
    "MemoryUnit", "PROJECT_DIR", "RUN_DIR", "SNIPPET_DIR", "SOURCE_ROOT_DIR",
    "VideoFormat", "WEIGHT_DIR",
]

from typing import TYPE_CHECKING

from mon import core
from mon.core.constant import *
from mon.coreml import factory

if TYPE_CHECKING:
    from mon.coreml.typing import ModelPhaseType, ReductionType


# region Factory

ACCELERATOR  = core.Factory(name="Accelerator")
CALLBACK     = core.Factory(name="Callback")
DATAMODULE   = core.Factory(name="DataModule")
DATASET      = core.Factory(name="Dataset")
LAYER        = core.Factory(name="Layer")
LOGGER       = core.Factory(name="Logger")
LOSS         = core.Factory(name="Loss")
LR_SCHEDULER = factory.LRSchedulerFactory(name="LRScheduler")
METRIC       = core.Factory(name="Metric")
MODEL        = core.Factory(name="Model")
OPTIMIZER    = factory.OptimizerFactory(name="Optimizer")
STRATEGY     = core.Factory(name="Strategy")
TRANSFORM    = core.Factory(name="Transform")

# endregion


# region Enum

class ModelPhase(core.Enum):
    """Model training phases."""
    
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
        """Return a dictionary mapping strings to enum."""
        return {
            "training" : cls.TRAINING,
            "testing"  : cls.TESTING,
            "inference": cls.INFERENCE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enum."""
        return {
            0: cls.TRAINING,
            1: cls.TESTING,
            2: cls.INFERENCE,
        }

    @classmethod
    def from_str(cls, value: str) -> ModelPhase:
        """Convert a string to an enum."""
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ModelPhase:
        """Convert an integer to an enum."""
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: ModelPhaseType) -> ModelPhase | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, ModelPhase):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class Reduction(core.Enum):
    """Tensor reduction options"""
    
    NONE         = "none"
    MEAN         = "mean"
    SUM          = "sum"
    WEIGHTED_SUM = "weighted_sum"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to enum."""
        return {
            "none"        : cls.NONE,
            "mean"        : cls.MEAN,
            "sum"         : cls.SUM,
            "weighted_sum": cls.WEIGHTED_SUM
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enum."""
        return {
            0: cls.NONE,
            1: cls.MEAN,
            2: cls.SUM,
            3: cls.WEIGHTED_SUM,
        }

    @classmethod
    def from_str(cls, value: str) -> Reduction:
        """Convert a string to an enum."""
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> Reduction:
        """Convert an integer to an enum."""
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: ReductionType) -> Reduction | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, Reduction):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None

# endregion
