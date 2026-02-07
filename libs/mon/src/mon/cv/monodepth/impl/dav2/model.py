#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth Anything V2 models and pre-trained weights.

This module provides the Depth Anything V2 definition and pre-trained weights.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

__all__ = [
    "DAV2",
    "DAV2_ViTB_Weights",
    "DAV2_ViTL_Weights",
    "DAV2_ViTS_Weights",
    "dav2_vitb",
    "dav2_vitl",
    "dav2_vits",
]

import sys

import torch

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
from mon.core.types import Weights, WeightsEnum, WeightsType

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]
extern_path  = current_dir / "extern" / "dav2"
if str(extern_path) not in sys.path:
    sys.path.append(str(extern_path))

try:
    # Now we can safely import from the original repository
    from depth_anything_v2 import dpt
except ImportError:
    raise ImportError(f"Failed to import 'depth_anything_v2' from the 'extern/dav2' directory.")


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class DAV2(nn.Module, nn.RegistrableMixin):
    """DAV2 model for monocular depth estimation.

    References:
        - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
          Depth Estimation," NeurIPS 2024.
        - https://github.com/DepthAnything/Depth-Anything-V2
    """

    arch     : str          = "dav2"
    name     : str          = None
    tasks    : list[Task]   = [Task.MONODEPTH]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name        : str,
        encoder     : str,
        features    : int,
        out_channels: list[int],
        use_bn      : bool                     = False,
        use_clstoken: bool                     = False,
        device      : torch.device | str | int = torch.device("cpu"),
        weights     : WeightsType  | None      = None,
        verbose     : bool                     = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            weights: Pre-trained weights to load. Defaults to None.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        self.verbose = verbose

        # Load the base model
        # if isinstance(weights, WeightsEnum):
        #     kwargs["num_classes"] = weights.num_classes

        device     = create_device(device)
        base_model = dpt.DepthAnythingV2(
            encoder      = encoder,
            features     = features,
            out_channels = out_channels,
            use_bn       = use_bn,
            use_clstoken = use_clstoken,
            device       = device,
            *args, **kwargs
        )

        if isinstance(weights, WeightsType):
            base_model.load_state_dict(weights.state_dict(weights_only=True))
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # Assign the base model
        self.model = base_model

    # --- Callable & Context Manager ---
    def forward(self, *args, **kwargs):
        """Forward the input through the network.

        Simply delegates the call to the underlying model.
        """
        return self.model(*args, **kwargs)


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="dav2_vits")
class DAV2_ViTS_Weights(WeightsEnum):

    DA_2K = Weights(
        path        = ZOO_DIR / "cv/monodepth/dav2/dav2_vits/da2k/dav2_vits_da2k.pth",
        url         = "",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = DA_2K


@WEIGHTS.register(name="dav2_vitb")
class DAV2_ViTB_Weights(WeightsEnum):

    DA_2K = Weights(
        path        = ZOO_DIR / "cv/monodepth/dav2/dav2_vitb/da2k/dav2_vitb_da2k.pth",
        url         = "",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = DA_2K


@WEIGHTS.register(name="dav2_vitl")
class DAV2_ViTL_Weights(WeightsEnum):

    DA_2K = Weights(
        path        = ZOO_DIR / "cv/monodepth/dav2/dav2_vitl/da2k/dav2_vitl_da2k.pth",
        url         = "",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = DA_2K


# --- Model Variants ---

@MODELS.register(name="dav2_vits", metaclass=DAV2)
def dav2_vits(weights: WeightsEnum | str | None = DAV2_ViTS_Weights.DEFAULT, *args, **kwargs):
    """Create an DAV2 model.

    Args:
        weights: Pre-trained weights to load. Defaults to DAV2_ViTS_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An DAV2 model instance.
    """
    return DAV2(
        name         = "dav2_vits",
        encoder      = "vits",
        features     = 64,
        out_channels = [48, 96, 192, 384],
        weights      = DAV2_ViTS_Weights(weights),
        *args, **kwargs
    )


@MODELS.register(name="dav2_vitb", metaclass=DAV2)
def dav2_vitb(weights: WeightsEnum | str | None = DAV2_ViTB_Weights.DEFAULT, *args, **kwargs):
    """Create an DAV2 model.

    Args:
        weights: Pre-trained weights to load. Defaults to DAV2_ViTB_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An DAV2 model instance.
    """
    return DAV2(
        name         = "dav2_vitb",
        encoder      = "vitb",
        features     = 128,
        out_channels = [96, 192, 384, 768],
        weights      = DAV2_ViTB_Weights(weights),
        *args, **kwargs
    )


@MODELS.register(name="dav2_vitl", metaclass=DAV2)
def dav2_vitl(weights: WeightsEnum | str | None = DAV2_ViTL_Weights.DEFAULT, *args, **kwargs):
    """Create an DAV2 model.

    Args:
        weights: Pre-trained weights to load. Defaults to DAV2_ViTL_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An DAV2 model instance.
    """
    return DAV2(
        name         = "dav2_vitl",
        encoder      = "vitl",
        features     = 256,
        out_channels = [256, 512, 1024, 1024],
        weights      = DAV2_ViTL_Weights(weights),
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
