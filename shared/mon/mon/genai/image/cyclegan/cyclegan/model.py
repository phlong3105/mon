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
]

import argparse
from typing import Any

import box

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .models.cycle_gan_model import CycleGANModel
from .models.pix2pix_model import Pix2PixModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="cyclegan", arch="cyclegan")
class CycleGAN(CycleGANModel, nn.ModelMixin):
    """CycleGAN model for image-to-image translation.
    
    References:
        - Paper: "Unpaired Image-to-Image Translation using Cycle-Consistent
          Adversarial Networks," ICCV 2017.
        - Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    
    _arch     : str          = "cyclegan"
    _name     : str          = "cyclegan"
    _tasks    : list[Task]   = [Task.IMG2IMG]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self, opt: argparse.Namespace, weights: Any = None):
        super().__init__(opt)
        # Load weights
        _, path, _ = self.parse_weights(weights)
        self.setup(path, opt)
        

@MODELS.register(name="pix2pix", arch="pix2pix")
class Pix2Pix(Pix2PixModel, nn.ModelMixin):
    """Pix2Pix model for image-to-image translation.
    
    References:
        - Paper: "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017.
        - Code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    
    _arch     : str          = "pix2pix"
    _name     : str          = "pix2pix"
    _tasks    : list[Task]   = [Task.IMG2IMG]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self, opt: argparse.Namespace, weights: Any = None):
        super().__init__(opt)
        # Load weights
        _, path, _ = self.parse_weights(weights)
        self.setup(path, opt)
