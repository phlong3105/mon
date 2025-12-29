#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss functions.

This package contains various loss functions commonly used in training machine
learning models, particularly in computer vision tasks.

Notes:
    - Design Pattern: Template Method.
    - Goal: Provide a structured way to define a family of methods or classes
      that share a common interface/inheritance but aren't tied to the specific
      "interchanged" requirement of the "Strategy Pattern".
    - Structure:
        ::
        
            template/
            ├── __init__.py    # Registry and factory logic
            ├── base.py        # Base classes and mixins
            ├── basic.py       # Basic functionalities
            ├── ...
            ├── utils.py       # Utility functions and helpers
            └── external/      # Expose external libraries
                └── ...
"""

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "BaseLoss",
    "CTCLoss",
    "CharbonnierLoss",
    "ColorConstancyLoss",
    "CosineEmbeddingLoss",
    "CosineSimilarityLoss",
    "CrossEntropyLoss",
    "DepthAwareIlluminationLoss",
    "EdgeLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "ExtendedL1Loss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "NLLLoss2d",
    "PSNRLoss",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "SpatialConsistencyLoss",
    "StructureTextureDecompositionLoss",
    "TotalVariationLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

from .base import BaseLoss
from .basic import (
    BCELoss,
    BCEWithLogitsLoss,
    CTCLoss,
    CharbonnierLoss,
    CosineEmbeddingLoss,
    CosineSimilarityLoss,
    CrossEntropyLoss,
    ExtendedL1Loss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MSELoss,
    MarginRankingLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    NLLLoss2d,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)
from .external import *
from .image import (
    ColorConstancyLoss,
    DepthAwareIlluminationLoss,
    EdgeLoss,
    ExposureControlLoss,
    ExposureValueControlLoss,
    PSNRLoss,
    SpatialConsistencyLoss,
    StructureTextureDecompositionLoss,
    TotalVariationLoss,
)
from .utils import *


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
