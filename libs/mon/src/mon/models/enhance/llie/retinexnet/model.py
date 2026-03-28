#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RetinexNet Models.

This module provides the RetinexNet definition and pre-trained weights.

References:
    - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
    - Code: https://github.com/aasharma90/RetinexNet_PyTorch
"""

from __future__ import annotations

__all__ = [
    "RetinexNet",
    "RetinexNet_Weights",
    "retinexnet",
]

from typing import override

import torch

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
from .module import DecomNet, EnhanceNet

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class RetinexNet(ModelRegisterMixin, EnhancementModel):
    """RetinexNet model for low-light image enhancement.

    References:
        - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
        - Code: https://github.com/aasharma90/RetinexNet_PyTorch
    """

    arch: str = "retinexnet"
    name: str = "retinexnet"
    tasks: list[Task] = [Task.LLIE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.decom_net = DecomNet()
        self.enhance_net = EnhanceNet()

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
    def forward_step(self, data: dict, *args, **kwargs) -> dict:
        """Forward the input through the network.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        image = data["image"]
        decom = data.get("decom", False)

        # Decomposition
        R, L = self.decom_net(image)
        if decom:
            return { "R": R, "L": L }

        # Relighting
        L_delta = self.enhance_net(R, L)
        L_delta_3 = torch.cat((L_delta, L_delta, L_delta), dim=1)

        # Reconstruction
        S = R * L_delta_3

        # Return final and intermediate results for debugging
        return {
            "enhanced": S,
            "R": R,
            "L": L,
            "L_delta": L_delta,
        }

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="retinexnet")
class RetinexNet_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/llie/retinexnet/retinexnet/lol_v1/retinexnet_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE_ME


# --- Model Variants ---

@MODELS.register(name="retinexnet", metaclass=RetinexNet)
def retinexnet(weights: WeightsLike = "default", *args, **kwargs):
    """Create a RetinexNet model.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "retinexnet")
    return RetinexNet(
        name="retinexnet",
        weights=RetinexNet_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
