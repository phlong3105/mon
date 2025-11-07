#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines the SemanticMask class for handling segmentation masks.

Common Tasks:
    - Define the Mask class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "SemanticMask",
]

from typing import Union

import cv2
import numpy as np
import torch

from mon.core.dtypes import image as I
from mon.core.pathlib import Path


class SemanticMask(I.Image):
    """Segmentation mask.

    Args:
        data: Input data as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
            Default: ``None``.
        path: Semantic mask file path. Default: ``None``.
        root: Root directory for the semantic mask. Default: ``None``.
        flags: OpenCV flag to read image. Default: ``cv2.IMREAD_COLOR_BGR``.
        cache: If ``True``, caches image in memory. Default: ``False``.
    """

    def __init__(
        self,
        data : Union[torch.Tensor, np.ndarray] = None,
        path : Path = None,
        root : Path = None,
        flags: int  = cv2.IMREAD_COLOR_BGR,
        cache: bool = False,
    ):
        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
