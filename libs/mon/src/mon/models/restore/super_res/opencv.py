#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common Upsamplers.

This module provides common upsampler classes and utilities for upsampling
operations.
"""

from __future__ import annotations

__all__ = [

]

from abc import ABC, abstractmethod
from typing import Any
from mon.core import (
    K,
    MODELS,
    Path,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from torch import nn, Tensor
from mon.nn import ModelRegisterMixin
from mon.core import DeviceLike, IntOrTuple2, Path, Size, sys_ctx, WeightsLike
from .base import ImageUpsampler

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASIC UPSAMPLING MODELS
# ==============================================================================

class OpenCVUpsampling(ModelRegisterMixin, ImageUpsampler):
    """OpenCV-based upsampling model."""

    arch: str = "opencv"
    name: str = "opencv"
    tasks: list[Task] = [Task.UPSAMPLE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,

        *args, **kwargs
    ):
        """Initialize a new instance."""
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
