#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements HOG (Histogram of Oriented Gradients) feature
extraction method.
"""

from __future__ import annotations

__all__ = [
    "HOGEmbedder",
]

import cv2
import numpy as np

from mon.globals import EMBEDDERS
from mon.vision.feature import base


# region HOGEmbedder

@EMBEDDERS.register(name="hog_embedder")
class HOGEmbedder(base.Embedder):
    """HOG (Histogram of Oriented Gradients) feature embedder.
    
    Args:
        win_size: The window size should be chosen based on the size of the
            objects being tracked. A smaller window size is suitable for
            tracking small objects, while a larger window size is suitable for
            larger objects. Default: (64, 128).
        block_size: The block size should be chosen based on the level of detail
            required for tracking. A larger block size can capture more global
            features, while a smaller block size can capture more local
            features. Default: (16, 16).
        block_stride: The block stride should be chosen based on the speed of
            the objects being tracked. A smaller block stride can provide more
            accurate tracking, but may also require more computation. Defaults
            to (8, 8).
        cell_size: The cell size should be chosen based on the texture and
            structure of the objects being tracked. A smaller cell size can
            capture more detailed features, while a larger cell size can capture
            more general features. Default: (8, 8).
        nbins: The number of orientation bins should be chosen based on the
            complexity of the gradient orientations in the images. More
            orientation bins can provide more detailed information about the
            orientations, but may also increase the dimensionality of the
            feature vector and require more computation. Default: 9.
        
    See Also:
        - :class:`mon.vision.model.embedding.base.Embedder`.
        - :class:`cv2.HOGDescriptor`.
    """
    
    def __init__(
        self,
        win_size    : list[int] = (64, 128),
        block_size  : float     = (16, 16),
        block_stride: int       = (8, 8),
        cell_size   : int       = (8, 8),
        nbins       : int       = 9,
        *args, **kwargs
    ):
        super().__init__()
        self.hog = cv2.HOGDescriptor(
            winSize     = win_size,
            blockSize   = block_size,
            blockStride = block_stride,
            cellSize    = cell_size,
            nbins       = nbins,
        )
        
    def embed(self, indexes: np.ndarray, images: np.ndarray) -> list[np.ndarray]:
        """Extract features in the images.

        Args:
            indexes: A list of image indexes.
            images: Images of shape NHWC.

        Returns:
           A 2-D list of feature vectors.
        """
        features = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hog  = self.hog.compute(gray)
            features.append(hog)
        return features
    
# endregion
