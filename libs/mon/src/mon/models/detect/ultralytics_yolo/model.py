#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics YOLOs Models.

This module provides the Ultralytics YOLOs definition and pre-trained weights.

References:
    - Code: https://docs.ultralytics.com/models/
"""

from __future__ import annotations

__all__ = [
    "YOLO",
]

from torch import nn

from mon.core import Path, Task, WeightsLike
from mon.nn import ModelRegisterMixin

try:
    import ultralytics
except ImportError:
    raise ImportError("Please install 'ultralytics' first.")

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class YOLO(ModelRegisterMixin, nn.Module):
    """Ultralytics YOLOs models for object detection.

    References:
        - Code: https://github.com/ultralytics/ultralytics
    """

    arch: str = "yolo"
    name: str = "yolo"
    tasks: list[Task] = [Task.CLASSIFY, Task.DETECT, Task.SEGMENT, Task.POSE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        # Ultralytics YOLO can be initialized with the weights path directly
        base_model = ultralytics.YOLO(model=str(weights.path))

        # Assign the base model
        self.model = base_model

    # --- Callable & Context Manager ---
    def forward(self, *args, **kwargs):
        """Forward the input through the network.

        Simply delegates the call to the underlying model.
        """
        return self.model(*args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
