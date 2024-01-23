#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ORB feature extraction method."""

from __future__ import annotations

__all__ = [
    "ORBEmbedder",
]

import cv2
import numpy as np

from mon import core
from mon.globals import EMBEDDERS
from mon.vision.feature import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region ORBEmbedder

@EMBEDDERS.register(name="orb_embedder")
class ORBEmbedder(base.Embedder):
    """ORB feature embedder.
    
    Args:
        num_features: The maximum number of features to retain.
        scale_factor: Pyramid decimation ratio, greater than 1. scaleFactor==2
            means the classical pyramid, where each next level has 4x fewer
            pixels than the previous, but such a big scale factor will degrade
            the feature matching scores dramatically. But, a too close to 1
            scale factor will mean that to cover a certain scale range, you will
            need more pyramid levels and so the speed will suffer.
        num_levels: The number of pyramid levels. The smallest level will have
            a linear size equal to the input_image_linear_size /pow(scaleFactor,
            nlevels - firstLevel).
        edge_threshold: This is the size of the border where the features are
            not detected. It should roughly match the patchSize parameter.
        first_level: The level of pyramid to put source image to. Previous
            layers are filled with upscale source image.
        wta_k: The number of points that produce each element of the oriented
            BRIEF descriptor.
        score_type: The default HARRIS_SCORE means that the Harris algorithm is
            used to rank features (the score is written to KeyPoint::score and
            is used to retain best :param:`num_features` features); FAST_SCORE
            is an alternative value of the parameter that produces slightly less
            stable keypoints, but it is a little faster to compute.
        patch_size: The size of the patch used by the oriented BRIEF descriptor.
            Of course, on smaller pyramid layers, the perceived image area
            covered by a feature will be larger.
        fast_threshold: The fast threshold.
    
    See Also:
        - :class:`mon.vision.feature.base.Embedder`.
    """
    
    def __init__(
        self,
        num_features  : int   = 500,
        scale_factor  : float = 1.2,
        num_levels    : int   = 8,
        edge_threshold: int   = 31,
        first_level   : int   = 0,
        wta_k         : int   = 2,
        score_type    : int   = cv2.ORB_HARRIS_SCORE,
        patch_size    : int   = 31,
        fast_threshold: int   = 20,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.orb = cv2.ORB_create(
            nfeatures     = num_features,
            scaleFactor   = scale_factor,
            nlevels       = num_levels,
            edgeThreshold = edge_threshold,
            firstLevel    = first_level,
            WTA_K         = wta_k,
            scoreType     = score_type,
            patchSize     = patch_size,
            fastThreshold = fast_threshold,
        )
        
    def embed(self, indexes: np.ndarray, images: np.ndarray) -> list[np.ndarray]:
        """Extract features in the images.

        Args:
            indexes: A :class:`list` of image indexes.
            images: Images of shape :math:`[N, H, W, C]`.

        Returns:
           A 2-D :class:`list` of feature vectors.
        """
        features = []
        for image in images:
            kp, des = self.orb.detectAndCompute(image, None)
            # Convert descriptors to feature vectors
            feature = np.array([des[j].flatten() for j in range(len(kp))])
            features.append(feature)
        return features
    
# endregion
