#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TV-Denoise Models.

This module provides the TV-Denoise definition and pre-trained weights.
"""

from __future__ import annotations

__all__ = [
    "TVDenoise",
]

from typing import Any, override

from mon.core import MODELS, Path, Task
from mon.models.restore.base import RestorationModel
from mon.nn import ModelRegisterMixin
from mon.ops import tv_denoise

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@MODELS.register(name="tv_denoise")
class TVDenoise(ModelRegisterMixin, RestorationModel):
    """TV-Denoise model for image denoising."""

    arch: str = "tv_denoise"
    name: str = "tv_denoise"
    tasks: list[Task] = [Task.DENOISE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
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
        super().__init__()

        # Assign attributes
        self.verbose = verbose
        self.weight = weight
        self.num_iter = num_iter

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Perform a single forward step of the model.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Output data dictionary.
        """
        image = data["image"]
        weight = data.get("weight", self.weight)
        num_iter = data.get("num_iter", self.num_iter)
        restored = tv_denoise(
            image=image,
            weight=weight,
            num_iter=num_iter,
        )

        # Return final and intermediate results for debugging
        return { "restored": restored }

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
