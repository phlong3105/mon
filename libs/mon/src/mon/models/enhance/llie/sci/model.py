#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SCI Models.

This module provides the SCI definition and pre-trained weights.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI

    - Paper: "Learning with Self-Calibrator for Fast and Robust Low-Light
      Image Enhancement," TPAMI 2025.
    - Code: https://github.com/vis-opt-group/SCI
"""

from __future__ import annotations

__all__ = [
    "SCI",
    "SCI_PP",
    "SCI_PP_Weights",
    "SCI_Weights",
    "sci",
    "sci_pp",
]

import torch
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
from .module import (
    CalibrateNetwork,
    CalibrateNetworkPP,
    EnhanceNetwork,
    EnhanceNetwork_Ha,
    EnhanceNetwork_Hb,
)

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SCI(ModelRegisterMixin, nn.Module):
    """SCI model for low-light image enhancement.

    References:
        - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
          CVPR 2022.
        - Code: https://github.com/vis-opt-group/SCI
    """

    arch: str = "sci"
    name: str = "sci"
    tasks: list[Task] = [Task.LLIE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        stage: int = 3,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            stage (int, optional): Number of enhancement stages. Defaults to 3.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.stage = stage

        # Define network
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)

        # Load weights
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor, inference: bool = True) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            inference (bool, optional): If True, run in inference mode.
                Defaults to True.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        x = image

        if inference:
            i = self.enhance(x)
            r = x / i
            r = torch.clamp(r, 0.0, 1.0)
            return {
                "enhanced": r,
                "illumination": i,
            }
        else:
            i_list, r_list, x_list, a_list = [], [], [], []
            for i in range(self.stage):
                x_list.append(x)
                i = self.enhance(x)
                r = x / i
                r = torch.clamp(r, 0, 1)
                att = self.calibrate(r)
                x = x + att
                i_list.append(i)
                r_list.append(r)
                a_list.append(torch.abs(att))
            return {
                "x_list": x_list,
                "i_list": i_list,
                "r_list": r_list,
                "a_list": a_list,
            }


class SCI_PP(ModelRegisterMixin, nn.Module):
    """SCI++ model for low-light image enhancement.

    References:
        - Paper: "Learning with Self-Calibrator for Fast and Robust Low-Light
          Image Enhancement," TPAMI 2025.
        - Code: https://github.com/vis-opt-group/SCI
    """

    arch: str = "sci"
    name: str = "sci++"
    tasks: list[Task] = [Task.LLIE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        stage: int = 3,
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            stage (int, optional): Number of enhancement stages. Defaults to 3.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.stage = stage

        # Define network
        self.ha = EnhanceNetwork_Ha(layers=1, channels=3)
        self.hb = EnhanceNetwork_Hb(layers=3, channels=16)
        self.calibrate = CalibrateNetworkPP(layers=3, channels=16)

        # Load weights
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor, inference: bool = True) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            inference (bool, optional): If True, run in inference mode.
                Defaults to True.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        x = image

        if inference:
            i = self.ha(x)
            r = x / i
            r = torch.clamp(r, 0.0, 1.0)
            return {
                "enhanced": r,
                "illumination": i,
            }
        else:
            i_list, r_list, x_list, a_list = [], [], [], []

            i = self.ha(x)
            r = x / i
            r = torch.clamp(r, 0, 1)
            i_list.append(i)
            r_list.append(r)
            x_list.append(x)

            for i in range(self.stage):
                x_list.append(i)
                att = self.calibrate(r)
                att_1 = self.hb(att)

                i = i + att + att_1
                r = x / i
                r = torch.clamp(r, 0, 1)

                i_list.append(i)
                r_list.append(r)
                a_list.append(torch.abs(att))
            return {
                "x_list": x_list,
                "i_list": i_list,
                "r_list": r_list,
                "a_list": a_list,
            }

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="sci")
class SCI_Weights(WeightsEnum):

    EASY = Weights(
        path=K.ZOO_ROOT / "enhance/sci/sci/pretrained/sci_easy.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    MEDIUM = Weights(
        path=K.ZOO_ROOT / "enhance/sci/sci/pretrained/sci_medium.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DIFFICULT = Weights(
        path=K.ZOO_ROOT / "enhance/sci/sci/pretrained/sci_difficult.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DEFAULT = MEDIUM


@WEIGHTS.register(name="sci++")
class SCI_PP_Weights(WeightsEnum):

    DEFAULT = Weights(
        path=K.ZOO_ROOT / "enhance/sci/sci++/pretrained/sci++_1_3500.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )


# --- Model Variants ---

@MODELS.register(name="sci", metaclass=SCI)
def sci(weights: WeightsLike = "default", *args, **kwargs):
    """Create an SCI model.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "sci")
    stage = kwargs.pop("stage", 3)
    return SCI(
        name="sci",
        stage=stage,
        weights=SCI_Weights(weights),
        *args, **kwargs,
    )



@MODELS.register(name="sci++", metaclass=SCI_PP)
def sci_pp(weights: WeightsLike = "default", *args, **kwargs):
    """Create an SCI++ model.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "sci++")
    stage = kwargs.pop("stage", 3)
    return SCI_PP(
        name="sci++",
        stage=stage,
        weights=SCI_PP_Weights(weights),
        *args, **kwargs,
    )
# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
