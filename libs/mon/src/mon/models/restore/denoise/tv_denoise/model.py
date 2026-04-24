#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TV-Denoise Models.

This module provides the TV-Denoise definition and pre-trained weights.
"""

from __future__ import annotations

__all__ = [
    "TVDenoise",
]

from typing import override

from tensordict import TensorDict
from torch import Tensor

from mon.core import MODELS, Path, Size, Strategy, Task
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from mon.ops import tv_denoise

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@MODELS.register(name="tv_denoise")
class TVDenoise(ModelRegisterMixin, Model):
    """TV-Denoise model for image denoising."""

    arch: str = "tv_denoise"
    name: str = "tv_denoise"
    tasks: list[Task] = [Task.DENOISE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"restored"}
    debug_keys: set = {}

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
        super().__init__()

        # Assign attributes
        self.verbose = verbose
        self.weight = weight
        self.num_iter = num_iter

    # --- Callable & Context Manager ---
    @override
    def forward(self, image: Tensor, *args, **kwargs) -> Tensor:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: The denoised image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(self, image: Tensor, *args, **kwargs) -> Tensor:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: The denoised image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # 1. Network forward
        restored = tv_denoise(image=image, weight=self.weight, num_iter=self.num_iter)

        # 2. Return final and intermediate results for debugging
        return restored

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
        imgsz = Size.from_value(imgsz)
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=[])
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
