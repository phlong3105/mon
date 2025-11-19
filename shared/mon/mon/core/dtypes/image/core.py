#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Image class for image data handling."""

__all__ = [
    "Image",
]

from typing import Union

import cv2
import numpy as np
import torch

from mon.core.dtypes.datapoint import BaseTensorOrArray
from mon.core.pathlib import Path


class Image(BaseTensorOrArray):
    """Image object.

    Args:
        data: An RGB image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
            Default: ``None``.
        path: Image file path. Default: ``None``.
        root: Root directory of the image (of a dataset). Default: ``None``.
        flags: OpenCV flag to read image. One of: ``cv2.IMREAD_UNCHANGED``,
            ``cv2.IMREAD_GRAYSCALE``, ``cv2.IMREAD_COLOR_BGR``, ``cv2.IMREAD_COLOR``,
            ``cv2.IMREAD_ANYDEPTH``, ``cv2.IMREAD_ANYCOLOR``, ``cv2.IMREAD_COLOR_RGB``.
            Default: ``cv2.IMREAD_COLOR``.
        cache: If ``True``, caches image in memory. Default: ``False``.
    """
    
    def __init__(
        self,
        data : Union[torch.Tensor, np.ndarray] = None,
        path : Path = None,
        root : Path = None,
        flags: int  = cv2.IMREAD_COLOR,
        cache: bool = False,
    ):
        if all(d is None for d in [data, path]):
            raise ValueError("Either [data] or [path] must be provided to initialize the Image object.")
        if data is not None:
            orig_shape = data.shape
        elif Path(path).is_image_file(exist=True):
            from mon.core.dtypes.image import io
            orig_shape = io.read_shape(path=path)
        else:
            orig_shape = None

        super().__init__(data=data, orig_shape=orig_shape)
        self._path = Path(path) if path is not None else None
        self._root = Path(root) if root is not None else None
        self.flags = flags
        self.cache = cache
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the underlying data tensor."""
        return self._orig_shape
    
    @property
    def path(self) -> Path:
        """Returns the image file path."""
        return self._path
    
    @property
    def root(self) -> Path:
        """Returns the root directory for the image."""
        return self._root
    
    @property
    def meta(self) -> dict:
        """Returns metadata about the image.

        Returns:
            A ``dict`` with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "path"      : self.path,
            "orig_shape": self.orig_shape,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, Path) else None,
        }
    
    def load(self, reload: bool = False) -> np.ndarray:
        """Loads an image into memory.

        Args:
            reload: If ``True``, reload the image even if already cached.
                Default: ``False``.

        Returns:
            An image as a ``numpy.ndarray`` of shape :math:`(H, W, C)` in
            range :math:`[0, 255]`.
        """
        # Return the image if it is already loaded and not reloading
        if not reload and self._data is not None:
            return self._data

        # Load the image
        from mon.core.dtypes.image.io import load
        image = load(self.path, self.flags)

        # Update the original shape of the image
        if self._orig_shape != image.shape:
            self._orig_shape = image.shape

        # Cache the image if needed
        self._data = image if self.cache else None
        return image
