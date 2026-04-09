#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Upsample Models.

This module provides the Upsample models definition and pre-trained weights.
"""

from __future__ import annotations

__all__ = [
    "GuidedFilterUpsample",
    "InterUpsample",
]

from typing import Any, override

import cv2
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import functional as F_tv, InterpolationMode

from mon.core import MODELS, Path, Size, Task, UPSAMPLERS
from mon.models.restore.super_res.base import SuperResolutionModel
from mon.nn import ModelRegisterMixin
from mon.ops import guided_filter_upsample

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@UPSAMPLERS.register(name="interpolation")
class InterUpsample(ModelRegisterMixin, SuperResolutionModel):
    """Interpolation model for super-resolution."""

    arch: str = "interpolation"
    name: str = "interpolation"
    tasks: list[Task] = [Task.SUPER_RES]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(self, verbose: bool = True, *args, **kwargs):
        """Initialize a new instance.

        Args:
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__()

        # Assign attributes
        self.verbose = verbose

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Perform a single forward step of the model.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Output data dictionary.
        """
        x_lr = data["x_lr"]
        imgsz: Size = data["imgsz"]

        # 1. Tensor / TorchVision
        if isinstance(x_lr, Tensor):
            _, c, _, _ = x_lr.shape
            is_depth = (c == 1)
            mode = InterpolationMode.NEAREST_EXACT if is_depth else InterpolationMode.BICUBIC
            x_hr = F_tv.resize(
                img=x_lr,
                size=list(imgsz.hw),
                interpolation=mode,
                antialias=False,
            )

        # 2. Numpy / OpenCV
        elif isinstance(x_lr, ndarray):
            _, _, c = x_lr.shape
            is_depth = (c == 1)
            mode = cv2.INTER_NEAREST if is_depth else cv2.INTER_CUBIC
            x_hr = cv2.resize(src=x_lr, dsize=imgsz.wh, interpolation=mode)

        # 3. Error: Unsupported type
        else:
            raise ValueError(
                f"Expected 'x_lr' to be a tensor or ndarray, "
                f"but got {type(x_lr).__name__}."
            )

        # Return final and intermediate results for debugging
        return { "x_hr": x_hr }


@MODELS.register(name="guided_filter_upsample")
@UPSAMPLERS.register(name="guided_filter")
class GuidedFilterUpsample(ModelRegisterMixin, SuperResolutionModel):
    """Guided Filter Upsample model for super-resolution tasks."""

    arch: str = "guided_filter"
    name: str = "guided_filter_upsample"
    tasks: list[Task] = [Task.SUPER_RES]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(self, radius: int = 1, verbose: bool = True, *args, **kwargs):
        """Initialize a new instance.

        Args:
            radius (int, optional): Radius for the guided filter. Defaults to 1.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__()

        # Assign attributes
        self.verbose = verbose
        self.radius = radius

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Perform a single forward step of the model.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Output data dictionary.
        """
        x_lr = data["x_lr"]
        y_hr = data["y_hr"]
        y_lr = data.get("y_lr", None)
        x_hr = guided_filter_upsample(x_lr=x_lr, y_lr=y_lr, y_hr=y_hr, r=self.radius)

        # Return final and intermediate results for debugging
        return { "x_hr": x_hr }

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
