#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UniK3D.

This module provides the UniK3D definition and pre-trained weights.

References:
    - Paper: "UniK3D: Universal Camera Monocular 3D Estimation," CVPR 2025.
    - Code: https://github.com/lpiccinelli-eth/UniK3D
"""

from __future__ import annotations

__all__ = [
    "UniK3D",
    "UniK3D_ViTB_Weights",
    "UniK3D_ViTL_Weights",
    "UniK3D_ViTS_Weights",
    "unik3d_vitb",
    "unik3d_vitl",
    "unik3d_vits",
]

import sys

from mon import nn
from mon.core import (
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
extern_path  = current_dir / "extern" / "unik3d"
if str(extern_path) not in sys.path:
    sys.path.append(str(extern_path))

try:
    # Now we can safely import from the original repository
    import unik3d
except ImportError:
    raise ImportError(f"Failed to import 'unik3d' from the 'extern/unik3d' directory.")


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class UniK3D(nn.Module, nn.RegistrableMixin):
    """UniK3D model for monocular depth estimation.

    References:
        - Paper: "UniK3D: Universal Camera Monocular 3D Estimation," CVPR 2025.
        - Code: https://github.com/lpiccinelli-eth/UniK3D
    """

    arch     : str          = "unik3d"
    name     : str          = None
    tasks    : list[Task]   = [Task.MONODEPTH]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name   : str,
        weights: WeightsType  | None = None,
        verbose: bool                = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            weights: Pre-trained weights to load. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the base model.
            **kwargs: Additional keyword arguments for the base model.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        self.verbose = verbose

        # Load the base model
        # UniK3D can be initialized with the weights path directly

        hf_model_name = name.replace("_", "-")
        base_model    = unik3d.UniK3D.from_pretrained(f"lpiccinelli/{hf_model_name}")
        # base_model    = unik3d.UniK3D.from_pretrained(model_id=str(weights.path))

        # Assign the base model
        self.model = base_model

    # --- Callable & Context Manager ---
    def forward(self, *args, **kwargs):
        """Forward the input through the network.

        Simply delegates the call to the underlying model.
        """
        return self.model.infer(*args, **kwargs)


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="unik3d_vits")
class UniK3D_ViTS_Weights(WeightsEnum):

    PRETRAINED = Weights(
        path        = ZOO_DIR / "cv/monodepth/unik3d/unik3d_vits/",
        url         = None,
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = PRETRAINED


@WEIGHTS.register(name="unik3d_vitb")
class UniK3D_ViTB_Weights(WeightsEnum):

    PRETRAINED = Weights(
        path        = ZOO_DIR / "cv/monodepth/unik3d/unik3d_vitb",
        url         = None,
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = PRETRAINED


@WEIGHTS.register(name="unik3d_vitl")
class UniK3D_ViTL_Weights(WeightsEnum):

    PRETRAINED = Weights(
        path        = ZOO_DIR / "cv/monodepth/unik3d/unik3d_vitl",
        url         = None,
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = PRETRAINED


# --- Model Variants ---

@MODELS.register(name="unik3d_vits", metaclass=UniK3D)
def unik3d_vits(weights: WeightsEnum | str | None = UniK3D_ViTS_Weights.DEFAULT, *args, **kwargs):
    """Create an UniK3D model.

    Args:
        weights: Pre-trained weights to load. Defaults to UniK3D_ViTS_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An UniK3D model instance.
    """
    return UniK3D(
        name    = "unik3d_vits",
        weights = UniK3D_ViTS_Weights(weights),
        *args, **kwargs
    )


@MODELS.register(name="unik3d_vitb", metaclass=UniK3D)
def unik3d_vitb(weights: WeightsEnum | str | None = UniK3D_ViTB_Weights.DEFAULT, *args, **kwargs):
    """Create an UniK3D model.

    Args:
        weights: Pre-trained weights to load. Defaults to UniK3D_ViT_B_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An UniK3D model instance.
    """
    return UniK3D(
        name    = "unik3d_vitb",
        weights = UniK3D_ViTB_Weights(weights),
        *args, **kwargs
    )


@MODELS.register(name="unik3d_vitl", metaclass=UniK3D)
def unik3d_vitl(weights: WeightsEnum | str | None = UniK3D_ViTL_Weights.DEFAULT, *args, **kwargs):
    """Create an UniK3D model.

    Args:
        weights: Pre-trained weights to load. Defaults to UniK3D_ViT_L_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An UniK3D model instance.
    """
    return UniK3D(
        name    = "unik3d_vitl",
        weights = UniK3D_ViTL_Weights(weights),
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = unik3d_vits()
    print(model_)

# endregion
