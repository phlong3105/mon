#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements OPE (Overlap Path Embedding) feature
extraction method.
"""

from __future__ import annotations

__all__ = [
    "OPEmbedder",
]

import torch

from mon.globals import EMBEDDERS
from mon.vision import core, nn
from mon.vision.feature import base

math         = core.math
console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region OPGEmbedder

@EMBEDDERS.register(name="op_embedder")
class OPEmbedder(base.Embedder, nn.Module):
    """HOG (Histogram of Oriented Gradients) feature embedder.
    
    Args:
        win_size: The window size should be chosen based on the size of the
            objects being tracked. A smaller window size is suitable for
            tracking small objects, while a larger window size is suitable for
            larger objects. Default: ``(64, 128)``.
        block_size: The block size should be chosen based on the level of detail
            required for tracking. A larger block size can capture more global
            features, while a smaller block size can capture more local
            features. Default: ``(16, 16)``.
        block_stride: The block stride should be chosen based on the speed of
            the objects being tracked. A smaller block stride can provide more
            accurate tracking, but may also require more computation. Defaults
            to ``(8, 8)``.
        cell_size: The cell size should be chosen based on the texture and
            structure of the objects being tracked. A smaller cell size can
            capture more detailed features, while a larger cell size can capture
            more general features. Default: ``(8, 8)``.
        nbins: The number of orientation bins should be chosen based on the
            complexity of the gradient orientations in the images. More
            orientation bins can provide more detailed information about the
            orientations, but may also increase the dimensionality of the
            feature vector and require more computation. Default: ``9``.
        
    See Also:
        - :class:`mon.vision.feature.base.Embedder`.
    """
    
    def __init__(
        self,
        image_size : int = 224,
        patch_size : int = 7,
        stride     : int = 4,
        in_channels: int = 3,
        nbins      : int = 9,
        embed_dim  : int = 768,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_size  = image_size
        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.proj        = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = embed_dim,
            kernel_size  = patch_size,
            stride       = stride,
            padding      = (patch_size  // 2, patch_size // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def embed(
        self,
        images: torch.Tensor,
        norm  : bool = False
    ) -> tuple[torch.Tensor, int, int]:
        """Extract features in the images.

        Args:
            indexes: A :class:`list` of image indexes.
            images: Images of shape :math:`[N, C, H, W]`.
            norm: Whether to normalize the features.

        Returns:
           A 2-D :class:`list` of feature vectors.
        """
        images     = self.proj(images)
        n, c, h, w = images.shape
        images     = images.flatten(2).transpose(1, 2)  # N, H, W, C
        if norm:
            images = self.norm(images)
        return images, h, w
    
# endregion
