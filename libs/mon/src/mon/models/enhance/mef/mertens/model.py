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

from typing import override

from tensordict import TensorDict
from torch import Tensor

from mon.core import MODELS, Path, Size, Strategy, Task
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from .module import mertens

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@MODELS.register(name="mertens")
class Mertens(ModelRegisterMixin, Model):
    """Mertens model for low-light image enhancement.

    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """

    arch: str = "mertens"
    name: str = "mertens"
    tasks: list[Task] = [Task.MEF]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"images"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {}

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
    @override
    def forward(self, images: Tensor | list[Tensor], *args, **kwargs) -> Tensor:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            images (Tensor | list[Tensor]): Input tensor of shape (B, C, H, W);
                or a list of images of shape (C, H, W).

        Returns:
            Tensor: Enhanced image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        return self.forward_step(images=images, *args, **kwargs)

    @override
    def forward_step(self, images: Tensor | list[Tensor], *args, **kwargs) -> Tensor:
        """Perform a single forward step of the model.

        Args:
            images (Tensor | list[Tensor]): Input tensor of shape (B, C, H, W);
                or a list of images of shape (C, H, W).

        Returns:
            Tensor: Enhanced image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # 1. Network forward
        enhanced = mertens(
            images=images,
            w_sat=self.w_sat,
            w_cont=self.w_cont,
            w_exp=self.w_exp,
            n_levels=self.n_levels
        )

        # 2. Return final and intermediate results for debugging
        return enhanced

    # --- Benchmark ---
    @override
    def benchmark(self, imgsz: Size, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (Size): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"images": dummy_input}, batch_size=[])
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

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
