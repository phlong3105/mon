#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DCE Models.

This module provides the Zero-DCE definition and pre-trained weights.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE

    - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
      Estimation," IEEE TPAMI 2022.
    - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE",
    "ZeroDCEPP",
    "ZeroDCEPP_Weights",
    "ZeroDCE_Weights",
    "zero_dce",
    "zero_dce_pp",
]

from typing import override

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
from .module import DSConv
from .utils import weights_init

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ZeroDCE(ModelRegisterMixin, Model):
    """Zero-DCE model for low-light image enhancement.

    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """

    arch: str = "zero_dce"
    name: str = "zero_dce"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"r"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "zero_dce",
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str, optional): Name of the model variant.
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            out_channels (int, optional): Number of output channels.
                Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define network
        self.e_conv1 = nn.Conv2d(in_channels, hidden_dim, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(hidden_dim * 2, out_channels * 8, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.apply(weights_init)

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
    ) -> tuple[Tensor, Tensor]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)

        return self.forward_step(image=image)

    @override
    def forward_step(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        # 1. Network forward
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # 2. Enhancement logic
        r_list = torch.split(r, 3, dim=1)
        y = image
        intermediates = {}

        for i, ri in enumerate(r_list):
            # Using y = y + ... is standard, but keeping track of
            # intermediates for debug is easier with a loop
            y = y + ri * (torch.pow(y, 2) - y)
            if i < len(r_list) - 1: # Don't add y8 to intermediates yet
                intermediates[f"y{i+1}"] = y

        # 3. Return final and intermediate results for debugging
        return y, r

    def forward_patch(
        self,
        image: Tensor,
        patcher: dict | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

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
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=[])
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)


class ZeroDCEPP(ModelRegisterMixin, Model):
    """Zero-DCE++ model for low-light image enhancement.

    References:
        - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
          Estimation," IEEE TPAMI 2022.
        - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
    """

    arch: str = "zero_dce"
    name: str = "zero_dce++"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"r"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "zero_dce++",
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        scale_factor: float = 1.0,
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
            scale_factor (float, optional): Upsampling scale factor.
                Defaults to 1.0.
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
        self.e_conv1 = DSConv(in_channels, hidden_dim)
        self.e_conv2 = DSConv(hidden_dim, hidden_dim)
        self.e_conv3 = DSConv(hidden_dim, hidden_dim)
        self.e_conv4 = DSConv(hidden_dim, hidden_dim)
        self.e_conv5 = DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv6 = DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv7 = DSConv(hidden_dim * 2, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

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
    ) -> tuple[Tensor, Tensor]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)

        return self.forward_step(image=image)

    @override
    def forward_step(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        # 1. Network forward with optional downsampling
        if self.scale_factor == 1:
            x_down = image
        else:
            x_down = F.interpolate(image, scale_factor=1 / self.scale_factor, mode="bilinear")

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if self.scale_factor == 1:
            r = r
        else:
            r = self.upsample(r)

        # 2 Enhancement logic
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
        patcher: dict | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

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
        data = TensorDict({"image": dummy_input}, batch_size=[])
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="zero_dce")
class ZeroDCE_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/lle/zero_dce/zero_dce/sice_me/zero_dce_sice_me.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DEFAULT = SICE_ME


@WEIGHTS.register(name="zero_dce++")
class ZeroDCEPP_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/lle/zero_dce/zero_dce++/sice_me/zero_dce++_sice_me.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DEFAULT = SICE_ME


# --- Model Variants ---

@MODELS.register(name="zero_dce", metaclass=ZeroDCE)
def zero_dce(weights: WeightsLike = "default", *args, **kwargs):
    """Create a Zero-DCE model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "zero_dce")
    in_channels = kwargs.pop("in_channels", 3)
    out_channels = kwargs.pop("out_channels", 3)
    hidden_dim = kwargs.pop("hidden_dim", 32)
    return ZeroDCE(
        name="zero_dce",
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        weights=ZeroDCE_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="zero_dce++", metaclass=ZeroDCEPP)
def zero_dce_pp(weights: WeightsLike = "default", *args, **kwargs):
    """Create a Zero-DCE++ model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "zero_dce++")
    in_channels = kwargs.pop("in_channels", 3)
    out_channels = kwargs.pop("out_channels", 3)
    hidden_dim = kwargs.pop("hidden_dim", 32)
    scale_factor = kwargs.pop("scale_factor", 1)
    return ZeroDCEPP(
        name="zero_dce++",
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        scale_factor=scale_factor,
        weights=ZeroDCEPP_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
