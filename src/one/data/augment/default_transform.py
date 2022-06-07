#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torch.nn
from torchvision import transforms

from one.core import TRANSFORMS

# MARK: - Register

# MARK: Vision Transforms
TRANSFORMS.register(name="center_crop",             module=transforms.CenterCrop)
TRANSFORMS.register(name="color_jitter",            module=transforms.ColorJitter)
TRANSFORMS.register(name="convert_image_dtype",     module=transforms.ConvertImageDtype)
TRANSFORMS.register(name="five_crop",               module=transforms.FiveCrop)
TRANSFORMS.register(name="gaussian_blur",           module=transforms.GaussianBlur)
TRANSFORMS.register(name="grayscale",               module=transforms.Grayscale)
TRANSFORMS.register(name="identity",                module=torch.nn.Identity)
TRANSFORMS.register(name="linear_transformation",   module=transforms.LinearTransformation)
TRANSFORMS.register(name="normalize",               module=transforms.Normalize)
TRANSFORMS.register(name="pad",                     module=transforms.Pad)
TRANSFORMS.register(name="pil_to_tensor",           module=transforms.PILToTensor)
TRANSFORMS.register(name="random_adjust_sharpness", module=transforms.RandomAdjustSharpness)
TRANSFORMS.register(name="random_affine",           module=transforms.RandomAffine)
TRANSFORMS.register(name="random_apply",            module=transforms.RandomApply)
TRANSFORMS.register(name="random_auto_contrast",    module=transforms.RandomAutocontrast)
TRANSFORMS.register(name="random_choice",           module=transforms.RandomChoice)
TRANSFORMS.register(name="random_crop",             module=transforms.RandomCrop)
TRANSFORMS.register(name="random_equalize",         module=transforms.RandomEqualize)
TRANSFORMS.register(name="random_erasing",          module=transforms.RandomErasing)
TRANSFORMS.register(name="random_grayscale",        module=transforms.RandomGrayscale)
TRANSFORMS.register(name="random_horizontal_flip",  module=transforms.RandomHorizontalFlip)
TRANSFORMS.register(name="random_invert",           module=transforms.RandomInvert)
TRANSFORMS.register(name="random_order",            module=transforms.RandomOrder)
TRANSFORMS.register(name="random_perspective",      module=transforms.RandomPerspective)
TRANSFORMS.register(name="random_posterize",        module=transforms.RandomPosterize)
TRANSFORMS.register(name="random_resized_crop",     module=transforms.RandomResizedCrop)
TRANSFORMS.register(name="random_rotation",         module=transforms.RandomRotation)
# TRANSFORMS.register(name="random_sized_crop", module=transforms.RandomSizedCrop)
TRANSFORMS.register(name="random_solarize",         module=transforms.RandomSolarize)
TRANSFORMS.register(name="random_vertical_flip",    module=transforms.RandomVerticalFlip)
TRANSFORMS.register(name="resize",                  module=transforms.Resize)
# TRANSFORMS.register(name="scale",            module=transforms.Scale)
TRANSFORMS.register(name="ten_crop",                module=transforms.TenCrop)
TRANSFORMS.register(name="to_pil_image",            module=transforms.ToPILImage)
# TRANSFORMS.register(name="to_tensor",        module=transforms.ToTensor)
