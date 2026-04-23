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
    Config,
    is_weights_type,
    K,
    log,
    MODELS,
    Path,
    Size,
    Strategy,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.dataset import transform as T
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
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
    def forward(self, data: TensorDict, use_patch: bool = False, *args, **kwargs) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.
            use_patch (bool, optional): Whether to use patch-based processing.
                Defaults to False.

        Returns:
            TensorDict: Output data dictionary.
        """
        if use_patch:
            return self.forward_patch(data, *args, **kwargs)

        return self.forward_step(data)

    def forward_step(self, data: TensorDict) -> TensorDict:
        """Forward the input through the network using the standard forward method.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        enhanced, r = self._forward_core(image=data["image"])
        outputs = {
            "enhanced": enhanced,
            "r": r,
        }
        return TensorDict(outputs, batch_size=[])

    def forward_patch(
        self,
        data: TensorDict,
        patch_size: int = 512,
        overlap: int = 128,
    ) -> TensorDict:
        """Forward the input through the network using patch-based processing.

        Args:
            data (TensorDict): Input data dictionary.
            patch_size (int, optional): Size of the patches to process.
                Defaults to 512.
            overlap (int, optional): Overlap between patches. Defaults to 128.

        Returns:
            TensorDict: Output data dictionary.
        """
        image = data["image"]
        b, c, h, w = image.shape
        stride = patch_size - overlap

        # 1. Pad the image so we don't drop the bottom/right edges
        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        # 2. Setup output buffers (accumulator and weight map for blending)
        enhanced = torch.zeros_like(image_padded)
        weight_mask = torch.zeros_like(image_padded)

        # 3. Create a 2D Hann Window to feather the edges of the patches
        window_1d = torch.hann_window(patch_size).to(image.device)
        window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)
        window = window_2d.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

        # 4. Sliding Window Loop
        for y in range(0, image_padded.shape[2] - patch_size + 1, stride):
            for x in range(0, image_padded.shape[3] - patch_size + 1, stride):
                # Extract the local patch
                patch = image_padded[:, :, y:y+patch_size, x:x+patch_size]

                # Forward pass through the SOTA model
                enhanced_patch, r_patch = self._forward_core(image=patch)

                # Accumulate the blended output
                enhanced[:, :, y:y+patch_size, x:x+patch_size] += enhanced_patch * window
                weight_mask[:, :, y:y+patch_size, x:x+patch_size] += window

        # 5. Normalize by the accumulated weights
        enhanced = enhanced / (weight_mask + 1e-8)

        # 6. Crop off the padding to return the original dimensions
        enhanced = enhanced[:, :, :h, :w]

        # 7. Return final and intermediate results for debugging
        outputs = {
            "enhanced": enhanced,
        }
        return TensorDict(outputs, batch_size=[])

    def _forward_core(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """The mathematical heart of the model (Shared by both strategies)."""
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
    def forward(self, data: TensorDict, use_patch: bool = False, *args, **kwargs) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.
            use_patch (bool, optional): Whether to use patch-based processing.
                Defaults to False.

        Returns:
            TensorDict: Output data dictionary.
        """
        if use_patch:
            return self.forward_patch(data, *args, **kwargs)

        return self.forward_step(data)

    def forward_step(self, data: TensorDict) -> TensorDict:
        """Forward the input through the network using the standard forward method.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        enhanced, r = self._forward_core(image=data["image"])
        outputs = {
            "enhanced": enhanced,
            "r": r,
        }
        return TensorDict(outputs, batch_size=[])

    def forward_patch(
        self,
        data: TensorDict,
        patch_size: int = 512,
        overlap: int = 128,
    ) -> TensorDict:
        """Forward the input through the network using patch-based processing.

        Args:
            data (TensorDict): Input data dictionary.
            patch_size (int, optional): Size of the patches to process.
                Defaults to 512.
            overlap (int, optional): Overlap between patches. Defaults to 128.

        Returns:
            TensorDict: Output data dictionary.
        """
        image = data["image"]
        b, c, h, w = image.shape
        stride = patch_size - overlap

        # 1. Pad the image so we don't drop the bottom/right edges
        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        # 2. Setup output buffers (accumulator and weight map for blending)
        enhanced = torch.zeros_like(image_padded)
        r = torch.zeros_like(image_padded)
        weight_mask = torch.zeros_like(image_padded)

        # 3. Create a 2D Hann Window to feather the edges of the patches
        window_1d = torch.hann_window(patch_size).to(image.device)
        window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)
        window = window_2d.unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1)

        # 4. Sliding Window Loop
        for y in range(0, image_padded.shape[2] - patch_size + 1, stride):
            for x in range(0, image_padded.shape[3] - patch_size + 1, stride):
                # Extract the local patch
                patch = image_padded[:, :, y:y+patch_size, x:x+patch_size]

                # Forward pass through the SOTA model
                enhanced_patch, r_patch = self._forward_core(image=patch)

                # Accumulate the blended output
                enhanced[:, :, y:y+patch_size, x:x+patch_size] += enhanced_patch * window
                r[:, :, y:y+patch_size, x:x+patch_size] += r_patch * window
                weight_mask[:, :, y:y+patch_size, x:x+patch_size] += window

        # 5. Normalize by the accumulated weights
        enhanced = enhanced / (weight_mask + 1e-8)
        r = r / (weight_mask + 1e-8)

        # 6. Crop off the padding to return the original dimensions
        enhanced = enhanced[:, :, :h, :w]
        r = r[:, :, :h, :w]

        # 7. Return final and intermediate results for debugging
        outputs = {
            "enhanced": enhanced,
            "r": r,
        }
        return TensorDict(outputs, batch_size=[])

    def _forward_core(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """The mathematical heart of the model (Shared by both strategies)."""
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
                imgsz = Size(height=imgsz.h // self.scale_factor, width=imgsz.w // self.scale_factor)
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
