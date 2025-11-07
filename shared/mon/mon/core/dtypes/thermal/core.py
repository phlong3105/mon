#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines the InfraredMap class for handling infrared thermal images.

Common Tasks:
    - Define the InfraredMap class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "InfraredMap",
]

from typing import Union

import cv2
import numpy as np
import torch

from mon.core.dtypes import image as I
from mon.core.enum import InfraredSource
from mon.core.pathlib import Path


class InfraredMap(I.Image):
    """Infrared map.

    Args:
        data: Input data as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
            Default: ``None``.
        path: Infrared map file path. Default: ``None``.
        root: Root directory for the infrared map. Default: ``None``.
        source: Source of infrared data. One of ``InfraredSource``.
            Default: ``InfraredSource.INFRARED``.
        flags: OpenCV flag to read image. Default: ``cv2.IMREAD_GRAYSCALE``.
        cache: If ``True``, caches image in memory. Default: ``False``.
    """

    def __init__(
        self,
        data  : Union[torch.Tensor, np.ndarray] = None,
        path  : Path           = None,
        root  : Path           = None,
        source: InfraredSource = InfraredSource.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        cache : bool           = False,
    ):
        source = InfraredSource.from_value(source)
        if source not in InfraredSource:
            raise ValueError(f"``source`` must be one of {InfraredSource}, got {source}.")

        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
        self.source = source
