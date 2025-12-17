#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements:
    - CycleGAN model for image-to-image translation.
    - Pix2Pix model for image-to-image translation.
    - Colorization model for image colorization (black & white image -> colorful images).

References:
    - Paper: "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017.
    - Paper: "Unpaired Image-to-Image Translation using Cycle-Consistent
      Adversarial Networks," ICCV 2017.
    - Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

__all__ = [
    "CycleGAN",
    "Pix2Pix",
    "TestOptions",
    "TrainOptions",
    "util",
]

from .model import CycleGAN, Pix2Pix
from .options import TestOptions, TrainOptions
