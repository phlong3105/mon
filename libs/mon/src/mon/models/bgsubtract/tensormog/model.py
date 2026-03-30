#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TensorMoG Models.

This module provides the TensorMoG definition and pre-trained weights.

References:
    - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic
      Scene Adaptation for Background Modeling," Sensors 2020.
"""

from __future__ import annotations

__all__ = [
    "TensorMOG",
    "tensormog",
]

from typing import Any, override

import torch

from mon.core import MODELS, Path, Task
from mon.models.bgsubtract.base import BackgroundSubtractionModel
from mon.nn import ModelRegisterMixin
from .module import HVR

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class TensorMOG(ModelRegisterMixin, BackgroundSubtractionModel):
    """TensorMoG model for background subtraction.

    References:
        - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic
          Scene Adaptation for Background Modeling," Sensors 2020.
    """

    arch: str = "tensormog"
    name: str = "tensormog"
    tasks: list[Task] = [Task.BGSUBTRACT]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        height: int = 512,
        width: int = 512,
        num_gaussians: int = 3,
        learning_rate: float = 0.02,
        matching_thres: float = 2 * 2,
        background_thres: float = 0.6,
        num_updates: int = 30,
        tau_rate: float = 0.01,
        tau_updating_rate: float = 0.025,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            window_size (int): Size of the local window for context aggregation.
            hidden_dim (int): Number of channels in the hidden layers.
            num_layers (int): Total number of layers in the network.
            add_layers (int): Number of additional layers for context aggregation.
            epochs (int, optional): Number of optimization epochs for
                single-image optimization. Defaults to 100.
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.device = device

        # Define network
        self.hvr = HVR(
            height=height,
            width=width,
            num_gaussians=num_gaussians,
            learning_rate=learning_rate,
            matching_thres=matching_thres,
            background_thres=background_thres,
            num_updates=num_updates,
            tau_rate=tau_rate,
            tau_updating_rate=tau_updating_rate,
            device=device,
        )

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Forward the input through the network.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        image = data["image"]
        self.hvr.update(image)
        background = self.hvr.get_background()
        foreground = self.hvr.get_foreground(image)
        return {
            "background": background,
            "foreground": foreground,
        }

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

@MODELS.register(name="tensormog", metaclass=TensorMOG)
def tensormog(*args, **kwargs):
    """Create a TensorMOG model."""
    _ = kwargs.pop("name", "tensormog")
    return TensorMOG(name="tensormog", *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
