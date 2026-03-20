#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DCC-Net Models.

This module provides the DCC-Net definition and pre-trained weights.

References:
    - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/Ian0926/DCC-Net
"""

from __future__ import annotations

__all__ = [
    "DCCNet",
    "DCCNet_Weights",
    "dccnet",
]

from torch import nn, Tensor

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
from mon.nn import ModelRegisterMixin
from .module import Net

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class DCCNet(ModelRegisterMixin, nn.Module):
    """DCC-Net model for low-light image enhancement.

    References:
        - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement,"
          CVPR 2022.
        - Code: https://github.com/Ian0926/DCC-Net
    """

    arch: str = "dccnet"
    name: str = "dccnet"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        d_hist: int = 64,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            d_hist (int, optional): Number of histogram bins for the C-Net.
                Defaults to 64.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.module = Net(d_hist=d_hist)

        # Load weights
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        enhanced, gray, color_hist = self.module(image)

        # Return final and intermediate results for debugging
        outputs = {
            "enhanced": enhanced,
            "gray": gray,
            "color_hist": color_hist,
        }
        return outputs

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="dccnet")
class DCCNet_Weights(WeightsEnum):

    SICE = Weights(
        path=K.ZOO_ROOT / "enhance/dccnet/dccnet/lol_v1/dccnet_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE


# --- Model Variants ---

@MODELS.register(name="dccnet", metaclass=DCCNet)
def dccnet(weights: WeightsLike = "default", *args, **kwargs):
    """Create a DCC-Net model.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "dccnet")
    d_hist = kwargs.pop("d_hist", 64)
    return DCCNet(
        name="dccnet",
        d_hist=d_hist,
        weights=DCCNet_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
