#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TV-Denoise Models.

This module provides the ZTV-Denoise definition and pre-trained weights.
"""

from __future__ import annotations

__all__ = [
    "TVDenoise",
]

from torch import nn, Tensor

from mon.core import Path, Task
from mon.cv.ops import tv_denoise
from mon.nn import ModelRegisterMixin

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class TVDenoise(ModelRegisterMixin, nn.Module):
    """TV-Denoise model for image denoising."""

    arch: str = "tv_denoise"
    name: str = "tv_denoise"
    tasks: list[Task] = [Task.RESTORE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weight: float = 0.1,
        num_iter: int = 50,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weight (float, optional): Weight of the denoised image. Defaults to 0.1.
            num_iter (int, optional): Number of iterations. Defaults to 50.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)
        # Initialize RegistrableMixin
        # ModelRegisterMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose = verbose
        self.weight = weight
        self.num_iter = num_iter

    # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        weight: float | None = None,
        num_iter: int | None = None,
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            weight (float, optional): Weight of the denoised image. Defaults to 0.1.
            num_iter (int, optional): Number of iterations. Defaults to None.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        restored = tv_denoise(
            image=image,
            weight=weight or self.weight,
            num_iter=num_iter or self.num_iter,
        )

        # Return final and intermediate results for debugging
        outputs = { "restored": restored }
        return outputs

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
