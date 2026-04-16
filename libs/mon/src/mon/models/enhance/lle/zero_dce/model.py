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
from torch import nn
from torch.nn import functional as F

from mon.core import (
    is_weights_type,
    K,
    log,
    MODELS,
    Path,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.models.enhance.base import EnhancementModel
from mon.nn import ModelRegisterMixin
from .module import DSConv
from .utils import weights_init

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ZeroDCE(ModelRegisterMixin, EnhancementModel):
    """Zero-DCE model for low-light image enhancement.

    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """

    arch: str = "zero_dce"
    name: str = "zero_dce"
    tasks: list[Task] = [Task.LLE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "zero_dce",
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        weights: WeightsLike | None = None,
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
            weights (WeightsLike, optional): Pre-trained weights to load.
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
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: TensorDict, *args, **kwargs) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        image = data["image"]

        # 2. Network forward
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # 3. Enhancement logic
        r_list = torch.split(r, 3, dim=1)
        y = image
        intermediates = {}

        for i, ri in enumerate(r_list):
            # Using y = y + ... is standard, but keeping track of
            # intermediates for debug is easier with a loop
            y = y + ri * (torch.pow(y, 2) - y)
            if i < len(r_list) - 1: # Don't add y8 to intermediates yet
                intermediates[f"y{i+1}"] = y

        # 4. Return final and intermediate results for debugging
        outputs = {
            "enhanced": y,
            "r": r,
            **intermediates
        }
        return TensorDict(outputs, batch_size=[])


class ZeroDCEPP(ModelRegisterMixin, EnhancementModel):
    """Zero-DCE++ model for low-light image enhancement.

    References:
        - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
          Estimation," IEEE TPAMI 2022.
        - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
    """

    arch: str = "zero_dce"
    name: str = "zero_dce++"
    tasks: list[Task] = [Task.LLE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "zero_dce++",
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        scale_factor: float = 1.0,
        weights: WeightsLike | None = None,
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
            weights (WeightsLike, optional): Pre-trained weights to load.
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
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: TensorDict, *args, **kwargs) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        image = data["image"]

        # 2. Network forward with optional downsampling
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

        # 3. Enhancement logic
        y = image
        intermediates = {}

        for i in range(8):
            # Using y = y + ... is standard, but keeping track of
            # intermediates for debug is easier with a loop
            y = y + r * (torch.pow(y, 2) - y)
            if i < 7: # Don't add y8 to intermediates yet
                intermediates[f"y{i+1}"] = y

        # 4. Return final and intermediate results for debugging
        outputs = {
            "enhanced": y,
            "r": r,
            **intermediates
        }
        return TensorDict(outputs, batch_size=[])

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
        meta={}
    )
    DEFAULT = SICE_ME


@WEIGHTS.register(name="zero_dce++")
class ZeroDCEPP_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/lle/zero_dce/zero_dce++/sice_me/zero_dce++_sice_me.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE_ME


# --- Model Variants ---

@MODELS.register(name="zero_dce", metaclass=ZeroDCE)
def zero_dce(weights: WeightsLike = "default", *args, **kwargs):
    """Create a Zero-DCE model.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
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
        weights (WeightsLike, optional): Pre-trained weights to load.
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
