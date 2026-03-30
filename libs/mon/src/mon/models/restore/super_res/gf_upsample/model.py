#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Guided Filter Upsample Models.

This module provides the Guided Filter Upsample definition and pre-trained weights.
"""

from __future__ import annotations

__all__ = [
    "GuidedFilterUpsample",
]

from typing import override

from mon.core import MODELS, Path, Task, UPSAMPLERS
from mon.models.restore.super_res.base import SuperResolutionModel
from mon.nn import ModelRegisterMixin
from mon.ops import guided_filter_upsample

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

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
    def forward_step(self, data: dict, *args, **kwargs) -> dict:
        """Perform a single forward step of the model.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        x_lr = data["x_lr"]
        y_hr = data["y_hr"]
        y_lr = data.get("y_lr", None)
        y_hr = guided_filter_upsample(x_lr=x_lr, y_lr=y_lr, y_hr=y_hr, r=self.radius)

        # Return final and intermediate results for debugging
        return { "y_hr": y_hr }

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
