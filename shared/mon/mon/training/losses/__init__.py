#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions.

In machine learning (ML), a loss function is used to measure model performance
by calculating the deviation of a model’s predictions from the correct,
“ground truth” predictions. Optimizing a model entails adjusting model parameters
to minimize the output of some loss function.

References:
    - https://www.ibm.com/think/topics/loss-function#1580786328
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

from .base import *
from .core import *
from .image import *
