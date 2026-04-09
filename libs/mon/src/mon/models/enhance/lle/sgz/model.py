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

from typing import Any, Literal, override

import torch
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
from .module import DSC, TC

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SGZ(ModelRegisterMixin, EnhancementModel):
    """SGZ model for low-light image enhancement.

    References:
        - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
          Enhancement," WACV 2022.
        - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
    """

    arch: str = "sgz"
    name: str = "sgz"
    tasks: list[Task] = [Task.LLE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "sgz",
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 32,
        scale_factor: int = 1,
        conv_type: Literal["tc", "dsc"] = "dsc",
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
            scale_factor (int, optional): The scale factor of the input image.
                If greater than 1, the input image will be downsampled by this
                factor before processing and the output will be upsampled back
                to the original size. Defaults to 1.0 (no scaling).
            conv_type (Literal["tc", "dsc"], optional): The type of convolution
                to use in the network. Must be one of "tc" (traditional
                convolution) or "dsc" (depthwise separable convolution). Defaults
                to "dsc".
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
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Forward the input through the network.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Output data dictionary.
        """
        image = data["image"]

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

        y0 = image
        y1 = y0 + r * (torch.pow(y0, 2) - y0)
        y2 = y1 + r * (torch.pow(y1, 2) - y1)
        y3 = y2 + r * (torch.pow(y2, 2) - y2)
        y4 = y3 + r * (torch.pow(y3, 2) - y3)
        y5 = y4 + r * (torch.pow(y4, 2) - y4)
        y6 = y5 + r * (torch.pow(y5, 2) - y5)
        y7 = y6 + r * (torch.pow(y6, 2) - y6)
        y8 = y7 + r * (torch.pow(y7, 2) - y7)

        # Return final and intermediate results for debugging
        return {
            "enhanced": y8,
            "r": r,
            "y1": y1,
            "y2": y2,
            "y3": y3,
            "y4": y4,
            "y5": y5,
            "y6": y6,
            "y7": y7,
        }

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
        weights (WeightsLike, optional): Pre-trained weights to load.
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
