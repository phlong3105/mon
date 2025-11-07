#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the DepthMap class for depth map data handling.

Common Tasks:
    - Define the DepthMap class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "DepthMap",
]

from typing import Union

import cv2
import numpy as np
import torch

from mon.core.dtypes import image as I
from mon.core.enum import DepthSource
from mon.core.pathlib import Path


class DepthMap(I.Image):
    """Depth map object.

    Args:
        data: Input data as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
            Default: ``None``.
        path: Depth map file path. Default: ``None``.
        root: Root directory for the depth map. Default: ``None``.
        source: Source of depth data. One of ``DepthSource``. Default: ``DepthSource.DAv2_ViTB``.
        flags: OpenCV flag to read image. Default: ``cv2.IMREAD_GRAYSCALE``.
        cache: If ``True``, caches image in memory. Default: ``False``.
    """

    def __init__(
        self,
        data  : Union[torch.Tensor, np.ndarray] = None,
        path  : Path        = None,
        root  : Path        = None,
        source: DepthSource = DepthSource.DAv2_ViTB,
        flags : int         = cv2.IMREAD_GRAYSCALE,
        cache : bool        = False,
    ):
        source = DepthSource.from_value(source)
        if source not in DepthSource:
            raise ValueError(f"``source`` must be one of {DepthSource}, got {source}.")

        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
        self.source = source
