#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SGZ Models.

This module provides the Zero-DCE definition and pre-trained weights.

References:
    - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
      Enhancement," WACV 2022.
    - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
"""

from __future__ import annotations

__all__ = [
    "SGZ",
    "SGZ_Weights",
    "sgz",
]

from typing import Literal, override

import torch
from tensordict import TensorDict
from torch import nn, Tensor
from torch.nn import functional as F

from mon.core import (
    is_weights_type,
    K,
    log,
    MODELS,
    PATCHERS,
    Path,
    Size,
    Strategy,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from mon.ops import ImagePatcher
from .module import DSC, TC

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SGZ(ModelRegisterMixin, Model):
    """SGZ model for low-light image enhancement.

    References:
        - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
          Enhancement," WACV 2022.
        - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
    """

    arch: str = "sgz"
    name: str = "sgz"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"r"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "sgz",
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        scale_factor: float = 1.0,
        conv_type: Literal["tc", "dsc"] = "dsc",
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            out_channels (int, optional): Number of output channels.
                Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            scale_factor (float, optional): The scale factor of the input image.
                If greater than 1, the input image will be downsampled by this
                factor before processing and the output will be upsampled back
                to the original size. Defaults to 1.0 (no scaling).
            conv_type (Literal["tc", "dsc"], optional): The type of convolution
                to use in the network. Must be one of "tc" (traditional
                convolution) or "dsc" (depthwise separable convolution). Defaults
                to "dsc".
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # Define network
        if conv_type == "dsc":
            conv = DSC
        elif conv_type == "tc":
            conv = TC
        else:
            raise ValueError(
                f"Unsupported conv type: {conv_type}. Must be one of: 'dsc', 'tc'."
            )

        # Zero-DCE DWC + p-shared
        self.e_conv1 = conv(in_channels, hidden_dim)
        self.e_conv2 = conv(hidden_dim, hidden_dim)
        self.e_conv3 = conv(hidden_dim, hidden_dim)
        self.e_conv4 = conv(hidden_dim, hidden_dim)
        self.e_conv5 = conv(hidden_dim * 2, hidden_dim)
        self.e_conv6 = conv(hidden_dim * 2, hidden_dim)
        self.e_conv7 = conv(hidden_dim * 2, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        # Load weights
        if weights is not None and is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        use_patch: bool = False,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)

        return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(self, image: Tensor, *args, **kwargs) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        # 1. Network forward with optional downsampling
        if self.scale_factor != 1:
            x_down = image
        else:
            x_down = F.interpolate(image, scale_factor=1.0 / self.scale_factor, mode="bilinear")

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor != 1:
            r = self.upsample(r)

        # 2. Enhancement logic
        y = image
        intermediates = {}

        for i in range(8):
            # Using y = y + ... is standard, but keeping track of
            # intermediates for debug is easier with a loop
            y = y + r * (torch.pow(y, 2) - y)
            if i < 7: # Don't add y8 to intermediates yet
                intermediates[f"y{i+1}"] = y

        # 3. Return final and intermediate results for debugging
        return y, r

    def forward_patch(
        self,
        image: Tensor,
        patcher: dict | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            enhanced, r = self.forward_step(image=patch)
            patch_output = {
                "enhanced": enhanced,
                "r": r,
            }
            # 2.2. Feed result back to Patcher
            patcher(patches=patch_output, x=x, y=y)

        # 3. Get the merged results
        return tuple(patcher.output.values())

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
        imgsz = Size(height=imgsz.h // self.scale_factor, width=imgsz.w // self.scale_factor)
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=dummy_input.shape)
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="sgz")
class SGZ_Weights(WeightsEnum):

    LOL_V1 = Weights(
        path=K.ZOO_ROOT / "enhance/lle/sgz/sgz/lol_v1/sgz_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = LOL_V1


# --- Model Variants ---

@MODELS.register(name="sgz", metaclass=SGZ)
def sgz(weights: WeightsLike = "default", *args, **kwargs):
    """Create a SGZ model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "sgz")
    in_channels = kwargs.pop("in_channels", 3)
    out_channels = kwargs.pop("out_channels", 3)
    hidden_dim = kwargs.pop("hidden_dim", 32)
    scale_factor = kwargs.pop("scale_factor", 1)
    conv_type = kwargs.pop("conv_type", "dsc")
    return SGZ(
        name="sgz",
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        scale_factor=scale_factor,
        conv_type=conv_type,
        weights=SGZ_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
