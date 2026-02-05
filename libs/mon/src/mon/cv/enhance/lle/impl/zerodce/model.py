#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DCE.

This module provides the Zero-DCE definition and pre-trained weights.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE",
    "ZeroDCE_Weights",
    "zerodce",
]

import sys

import torch
import torch.nn.functional as F

from mon import nn
from mon.core import (
    create_device,
    log,
    MLType,
    MODELS,
    Path,
    Task,
    WEIGHTS,
    ZOO_DIR,
)
from mon.core.dtypes import Weights, WeightsEnum, WeightsType

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import zerodce' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m zerodce.predict
    from .utils import weights_init
except ImportError:
    # Works when running as a script: python predict.py
    from utils import weights_init


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class ZeroDCE(nn.Module, nn.RegistrableMixin):
    """ZeroDCE model for low-light image enhancement.

    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """

    arch     : str          = "zerodce"
    name     : str          = "zerodce"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name        : str,
        in_channels : int  = 3,
        out_channels: int  = 24,
        hidden_dim  : int  = 32,
        weights     : WeightsType | None = None,
        device      : torch.device = torch.device("cpu"),
        verbose     : bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            in_channels: Number of input channels. Defaults to 3.
            out_channels: Number of output channels. Defaults to 24.
            hidden_dim: Hidden dimension. Defaults to 32.
            weights: Pre-trained weights to load.
            device: Device to use for computation. Defaults to "cpu".
            verbose: Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments for the model.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose      = verbose
        self.in_channels  = in_channels
        self.hidden_dim   = hidden_dim
        self.out_channels = out_channels
        self.device      = create_device(device)

        # Define network
        self.e_conv1  = nn.Conv2d(in_channels,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim * 2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim * 2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim * 2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.apply(weights_init)

        # Load weights
        if isinstance(weights, WeightsType):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor, save_debug: bool = False) -> dict:
        """Forward the input through the network.

        Args:
            image: Image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and values ranging from 0.0 to 1.0.
            save_debug: Whether to save intermediate results for debugging.
                Defaults to False.
        """
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(r, 3, dim=1)
        y0 = image
        y1 = y0 + r1 * (torch.pow(y0, 2) - y0)
        y2 = y1 + r2 * (torch.pow(y1, 2) - y1)
        y3 = y2 + r3 * (torch.pow(y2, 2) - y2)
        y4 = y3 + r4 * (torch.pow(y3, 2) - y3)
        y5 = y4 + r5 * (torch.pow(y4, 2) - y4)
        y6 = y5 + r6 * (torch.pow(y5, 2) - y5)
        y7 = y6 + r7 * (torch.pow(y6, 2) - y6)
        y8 = y7 + r8 * (torch.pow(y7, 2) - y7)

        # 8. Return final and intermediate results for debugging
        outputs = { "enhanced": y8 }
        if save_debug:
            outputs |= {
                "r" : r,
                "y1": y1,
                "y2": y2,
                "y3": y3,
                "y4": y4,
                "y5": y5,
                "y6": y6,
                "y7": y7,
            }
        return outputs


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="zerodce")
class ZeroDCE_Weights(WeightsEnum):

    SICEME = Weights(
        path        = ZOO_DIR / "cv/enhance/lle/zerodce/zerodce/siceme/zerodce_siceme.pth",
        url         = "",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SICEME


# --- Model Variants ---

@MODELS.register(name="zerodce", metaclass=ZeroDCE)
def zerodce(weights: WeightsEnum | str | None = ZeroDCE_Weights.DEFAULT, *args, **kwargs):
    """Create a Zero-DCE model.

    Args:
        weights: Pre-trained weights to load. Defaults to ZeroDCE_Weights.DEFAULT.
        args: Additional positional arguments for the Zero-DCE model.
        kwargs: Additional keyword arguments for the Zero-DCE model.

    Returns:
        An Zero-DCE model instance.
    """
    return ZeroDCE(
        name         = "zerodce",
        in_channels  = 3,
        out_channels = 24,
        hidden_dim   = 32,
        weights      = ZeroDCE_Weights(weights),
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = zerodce()
    print(model_)

# endregion
