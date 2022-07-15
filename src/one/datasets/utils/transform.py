#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transforming and augmenting.
Taxonomy of transforms:

Vision
  |__ Automatic Augmentation
  |     |__ AutoAugment
  |     |__ RandAugment
  |     |__ TrivialAugmentWide
  |     |__ AugMix
  |
  |__ Compositions of transforms
  |     |__ Compose
  |
  |__ Conversion
  |     |__ PILToTensor
  |     |__ ToPILImage
  |     |__ ToTensor
  |
  |__ PIL Image and Tensor
  |     |__ CenterCrop
  |     |__ ColorJitter
  |     |__ FiveCrop
  |     |__ GaussianBlur
  |     |__ Grayscale
  |     |__ Pad
  |     |__ RandomAdjustSharpness
  |     |__ RandomAffine
  |     |__ RandomApply
  |     |__ RandomCrop
  |     |__ RandomEqualize
  |     |__ RandomGrayscale
  |     |__ RandomHorizontalFlip
  |     |__ RandomInvert
  |     |__ RandomPerspective
  |     |__ RandomPosterize
  |     |__ RandomResizedCrop
  |     |__ RandomRotation
  |     |__ RandomSolarize
  |     |__ RandomVerticalFlip
  |     |__ Resize
  |     |__ RandomInvert
  |     |__ TenCrop
  |
  |__ Tensor
        |__ LinearTransformation
        |__ Normalize
        |__ RandomErasing
        |__ ConvertImageDtype
"""

from __future__ import annotations

import inspect
import sys

# MARK: - Automatic Augmentation


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
