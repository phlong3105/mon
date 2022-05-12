#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MBLLEN: Multi-Branch Low-Light Enhancement Network. Fkey idea is to
extract rich features up to different levels, so that we can apply enhancement
via multiple subnets and finally produce the output image via multi-branch
fusion. In this manner, image quality is improved from different aspects.

References:
    https://github.com/Lvfeifan/MBLLEN
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import Indexes
from one.core import Int2T
from one.core import MODELS
from one.core import Pretrained
from one.core import to_2tuple
from one.nn import ConvReLU
from one.vision.enhancement.image_enhancer import ImageEnhancer
from one.vision.enhancement.mbllen.loss import MBLLENLoss

__all__ = [
    "MBLLEN",
]


# MARK: - EM

class EM(nn.Module):
    """FEnhancement regression (EM) has a symmetric structure to first apply
    convolutions and then deconvolutions.

    Attributes:
        channels (int:
            Number of channels for `Conv2D` layers used in each EM block.
            Default: `8`.
        kernel_size (Int2T):
            Kernel size for `Conv2D` layers used in each EM block.
            Default: `5`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, channels: int = 8, kernel_size: Int2T = 5):
        super().__init__()
        self.channels    = channels
        self.kernel_size = to_2tuple(kernel_size)
        
        self.convolutions = nn.Sequential(
            nn.Conv2d(32, self.channels, kernel_size=(3, 3), padding=1,
                      padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels * 2, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 4, self.kernel_size),
            nn.ReLU()
        )
        self.deconvolutions = nn.Sequential(
            nn.ConvTranspose2d(self.channels * 4, self.channels * 2,
                               self.kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels * 2, self.channels,
                               self.kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels, 3, self.kernel_size),
            nn.ReLU(),
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.convolutions(x)
        x = self.deconvolutions(x)
        return x
    

# MARK: - MBLLEN

@MODELS.register(name="mbllen")
class MBLLEN(ImageEnhancer):
    """MBLLEN consists of three modules: the feature extraction regression
    (FEM), the enhancement regression (EM) and the fusion regression (FM).
    
    References:
        https://github.com/Lvfeifan/MBLLEN
    
    Args:
        name (str, optional):
            Name of the backbone. Default: `mbllen`.
        out_indexes (Indexes):
            List of output tensors taken from specific layers' indexes.
            If `>= 0`, return the ith layer's output.
            If `-1`, return the final layer's output. Default: `-1`.
        pretrained (Pretrained):
            Use pretrained weights. If `True`, returns a model pre-trained on
            ImageNet. If `str`, load weights from saved file. Default: `True`.
            - If `True`, returns a model pre-trained on ImageNet.
            - If `str` and is a weight file(path), then load weights from
              saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
  
    Notes:
        - MBLLEN model requires input shape to be [:, 256, 256].
        - Optimizer should be: dict(name="adam", lr=0.0001)
    """

    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        channels   : int           = 8,
        kernel_size: int           = 5,
        num_blocks : int           = 10,
        # BaseModel's args
        basename   : Optional[str] = "mbllen",
        name       : Optional[str] = "mbllen",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs["loss"] = MBLLENLoss()
        super().__init__(
            basename    = basename,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        # NOTE: Get Hyperparameters
        self.channels    = channels
        self.kernel_size = to_2tuple(kernel_size)
        self.num_blocks  = num_blocks

        # NOTE: Features
        fem      = [nn.Conv2d(3, 32, (3, 3), padding=1, padding_mode="replicate")]
        fem     += [ConvReLU(32, 32, (3, 3), padding=1, padding_mode="replicate")
                    for _ in range(self.num_blocks - 1)]
        self.fem = nn.ModuleList(fem)
    
        em       = [EM(self.channels, self.kernel_size) for _ in range(self.num_blocks)]
        self.em  = nn.ModuleList(em)
        
        fm       = nn.Conv2d(30, 3, (1, 1), padding=0, padding_mode="replicate")
        self.fm  = fm
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        
    # MARK: Forward Pass

    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass once. Implement the logic for a single forward pass.

		Args:
			x (Tensor):
				Input of shape [B, C, H, W].

		Returns:
			yhat (Tensor):
				Predictions.
		"""
        fem_feat  = x
        em_concat = None
        for idx, (fem_i, em_i) in enumerate(zip(self.fem, self.em)):
            fem_feat  = fem_i(fem_feat)
            em_feat   = em_i(fem_feat)
            em_concat = (em_feat if (em_concat is None)
                         else torch.cat(tensors=(em_concat, em_feat), dim=1))
        x = self.fm(em_concat)
        return x
    
    # MARK: Training
    
    def on_fit_start(self):
        """Called at the very beginning of fit."""
        super().on_fit_start()
        if self.shape:
            h, w, c = self.shape
            if h != 256 or w != 256:
                raise ValueError(
                    f"MBLLEN model requires image's shape to be [256, 256, :]. "
                    f"Got: {self.shape}."
                )
