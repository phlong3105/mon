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
]

from typing import TYPE_CHECKING

from mon import foundation
from mon.coreml import factory

if TYPE_CHECKING:
    from mon.coreml.typing import ModelPhaseType, ReductionType


# region Factory

ACCELERATOR  = foundation.Factory(name="Accelerator")
CALLBACK     = foundation.Factory(name="Callback")
DATAMODULE   = foundation.Factory(name="DataModule")
DATASET      = foundation.Factory(name="Dataset")
LAYER        = foundation.Factory(name="Layer")
LOGGER       = foundation.Factory(name="Logger")
LOSS         = foundation.Factory(name="Loss")
LR_SCHEDULER = factory.LRSchedulerFactory(name="LRScheduler")
METRIC       = foundation.Factory(name="Metric")
MODEL        = foundation.Factory(name="Model")
OPTIMIZER    = factory.OptimizerFactory(name="Optimizer")
STRATEGY     = foundation.Factory(name="Strategy")
TRANSFORM    = foundation.Factory(name="Transform")

# endregion


# region Enum

class ModelPhase(foundation.Enum):
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


class Reduction(foundation.Enum):
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
