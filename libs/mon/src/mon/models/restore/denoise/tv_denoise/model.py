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

from mon.core import Config, MODELS, Path, Size, Strategy, Task
from mon.dataset import transform as T
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
    def forward(self, data: TensorDict) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        image = data["image"]

        # 2. Network forward
        restored = tv_denoise(image=image, weight=self.weight, num_iter=self.num_iter)

        # 3. Return final and intermediate results for debugging
        outputs = {
            "restored": restored,
        }
        return TensorDict(outputs, batch_size=[])

    # --- Interfaces ---
    @override
    def build_transforms(self, config: Config | None = None) -> T.Compose:
        """Define the model's transformations.

        Args:
            config (Config, optional): The configuration object containing any
                necessary parameters for defining the transformations.
                Defaults to None.

        Returns:
            Callable: A callable (e.g., a torchvision transform or a custom
                function) that takes in the raw input data and returns the
                transformed data ready for the forward step.
        """
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

        if config is not None:
            if config.strategy in [Strategy.RESIZE]:
                imgsz = config.imgsz
                resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
                transforms = resize + transforms

        return transforms

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
