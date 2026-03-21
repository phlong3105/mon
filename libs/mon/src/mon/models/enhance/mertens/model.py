#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mertens Models.

This module provides the Mertens et al. Exposure Fusion model definition and
pre-trained weights.

References:
    - Paper: "Exposure Fusion," PG 2007.
    - Code: https://github.com/Jamy-L/Pytorch-Exposure-Fusion
"""

from __future__ import annotations

__all__ = [
    "Mertens",
]

from numpy import ndarray
from torch import nn, Tensor

from mon.core import MODELS, Path, Task
from mon.nn import ModelRegisterMixin
from .module import mertens

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@MODELS.register(name="mertens")
class Mertens(ModelRegisterMixin, nn.Module):
    """Mertens model for low-light image enhancement.

    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """

    arch: str = "mertens"
    name: str = "mertens"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        w_sat: float = 1.0,
        w_cont: float = 1.0,
        w_exp: float = 1.0,
        n_levels: int = 4,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            w_sat (float, optional): The saturation importance weight. Defaults to 1.0.
            w_cont (float, optional): The contrast importance weight. Defaults to 1.0.
            w_exp (float, optional): The well-exposed importance weight. Defaults to 1.0.
            n_levels (int, optional): The number of levels in the pyramids. Defaults to 4.
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.w_sat = w_sat
        self.w_cont = w_cont
        self.w_exp = w_exp
        self.n_levels = n_levels

    # --- Callable & Context Manager ---
    def forward(self, images: Tensor | list[Tensor] | list[ndarray]) -> dict:
        """Forward the input through the network.

        Args:
            images (Tensor | list[Tensor] | list[ndarray]): Input tensor of
                shape (B, C, H, W); or a list of images of shape (C, H, W).

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        enhanced = mertens(
            images=images,
            w_sat=self.w_sat,
            w_cont=self.w_cont,
            w_exp=self.w_exp,
            n_levels=self.n_levels
        )
        return {"enhanced": enhanced}

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
