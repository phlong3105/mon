#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics SAM models and pre-trained weights.

This module provides wrappers for Ultralytics SAM models and pre-trained weights.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

__all__ = [
    "SAM",
    "SAM2_1_B_Weights",
    "SAM2_1_L_Weights",
    "SAM2_1_S_Weights",
    "SAM2_1_T_Weights",
    "SAM2_B_Weights",
    "SAM2_L_Weights",
    "SAM2_S_Weights",
    "SAM2_T_Weights",
    "SAM3_Weights",
    "SAM_B_Weights",
    "SAM_L_Weights",
    "sam2_1_b",
    "sam2_1_l",
    "sam2_1_s",
    "sam2_1_t",
    "sam2_b",
    "sam2_l",
    "sam2_s",
    "sam2_t",
    "sam3",
    "sam_b",
    "sam_l",
]

from mon import nn
from mon.core import MLType, MODELS, Path, Task, WEIGHTS, ZOO_DIR
from mon.core.types import Weights, WeightsEnum, WeightsType

try:
    import ultralytics
except ImportError:
    raise ImportError("Please install 'ultralytics' first.")


current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class SAM(nn.Module, nn.RegistrableMixin):
    """Ultralytics SAM model for segmentation.

    References:
        - Code: https://github.com/ultralytics/ultralytics
    """

    arch     : str          = "sam"
    name     : str          = None
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name   : str,
        weights: WeightsType | None = None,
        verbose: bool               = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            weights: Pre-trained weights to load. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the SAM model.
            **kwargs: Additional keyword arguments for the SAM model.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        self.verbose = verbose

        # Load the base model
        # if isinstance(weights, WeightsType):
        #     kwargs["num_classes"] = weights.num_classes

        # Ultralytics SAM can be initialized with the weights path directly
        base_model = ultralytics.SAM(model=str(weights.path))

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

@WEIGHTS.register(name="sam_b")
class SAM_B_Weights(WeightsEnum):

    SA_1B = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam/sam_b/sa1b/sam_b_sa1b.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_1B


@WEIGHTS.register(name="sam_l")
class SAM_L_Weights(WeightsEnum):

    SA_1B = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam/sam_l/sa1b/sam_l_sa1b.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_l.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_1B


@WEIGHTS.register(name="sam2_t")
class SAM2_T_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2_t/sav/sam2_t_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2_s")
class SAM2_S_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2_s/sav/sam2_s_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_s.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2_b")
class SAM2_B_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2_b/sav/sam2_b_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2_l")
class SAM2_L_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2_l/sav/sam2_l_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_l.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2.1_t")
class SAM2_1_T_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2.1_t/sav/sam2.1_t_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2.1_s")
class SAM2_1_S_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2.1_s/sav/sam2.1_s_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2.1_b")
class SAM2_1_B_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2.1_b/sav/sam2.1_b_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam2.1_l")
class SAM2_1_L_Weights(WeightsEnum):

    SA_V = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam2/sam2.1_l/sav/sam2.1_l_sav.pt",
        url         = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_V


@WEIGHTS.register(name="sam3")
class SAM3_Weights(WeightsEnum):

    SA_CO = Weights(
        path        = ZOO_DIR / "cv/ultralytics/sam3/sam3/saco/sam3_saco.pt",
        url         = "https://huggingface.co/facebook/sam3/blob/main/sam3.pt",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = SA_CO


# --- Model Variants ---

@MODELS.register(name="sam_b", metaclass=SAM)
def sam_b(weights: WeightsEnum | str | None = SAM_B_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM_B_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam_b", weights=SAM_B_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam_l", metaclass=SAM)
def sam_l(weights: WeightsEnum | str | None = SAM_L_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM_L_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam_l", weights=SAM_L_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2_t", metaclass=SAM)
def sam2_t(weights: WeightsEnum | str | None = SAM2_T_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_T_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2_t", weights=SAM2_T_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2_s", metaclass=SAM)
def sam2_s(weights: WeightsEnum | str | None = SAM2_S_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_S_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2_s", weights=SAM2_S_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2_b", metaclass=SAM)
def sam2_b(weights: WeightsEnum | str | None = SAM2_B_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_B_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2_b", weights=SAM2_B_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2_l", metaclass=SAM)
def sam2_l(weights: WeightsEnum | str | None = SAM2_L_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_L_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2_l", weights=SAM2_L_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2.1_t", metaclass=SAM)
def sam2_1_t(weights: WeightsEnum | str | None = SAM2_1_L_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_1_L_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2.1_t", weights=SAM2_1_T_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2.1_s", metaclass=SAM)
def sam2_1_s(weights: WeightsEnum | str | None = SAM2_1_S_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_1_S_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2.1_s", weights=SAM2_1_S_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2.1_b", metaclass=SAM)
def sam2_1_b(weights: WeightsEnum | str | None = SAM2_1_B_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_1_B_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2.1_b", weights=SAM2_1_B_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam2.1_l", metaclass=SAM)
def sam2_1_l(weights: WeightsEnum | str | None = SAM2_1_L_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM2_1_L_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam2.1_l", weights=SAM2_1_L_Weights(weights),*args, **kwargs)


@MODELS.register(name="sam3", metaclass=SAM)
def sam3(weights: WeightsEnum | str | None = SAM3_Weights.DEFAULT, *args, **kwargs):
    """Create an SAM model.

    Args:
        weights: Pre-trained weights to load. Defaults to SAM3_Weights.DEFAULT.
        args: Additional positional arguments for the SAM model.
        kwargs: Additional keyword arguments for the SAM model.

    Returns:
        An SAM model instance.
    """
    return SAM(name="sam3", weights=SAM3_Weights(weights),*args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    # sam3_ = ultralytics.SAM(model="sam3.pt")
    pass

# endregion
