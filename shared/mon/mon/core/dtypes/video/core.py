#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the data structure for video frames."""

__all__ = [
    "Frame",
]

from typing import Union

import numpy as np
import torch

from mon.core.constants import SAVE_IMAGE_EXT
from mon.core.pathlib import Path
from ..base import BaseTensorOrArray


class Frame(BaseTensorOrArray):
    """Frame.

    Args:
        data: Input data as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
            Default: ``None``.
        index: Index of frame in video.
        orig_shape: Original shape of the image as a ``tuple`` of :math:`(H, W, C)`.
            Default: ``None``.
        path: Video file path. Default: ``None``.
        root: Root directory for the video. Default: ``None``.
    """

    def __init__(
        self,
        data      : Union[torch.Tensor, np.ndarray],
        index     : int,
        orig_shape: tuple[int, int, int] = None,
        path      : Path = None,
        root      : Path = None,
    ):
        if orig_shape is None:
            orig_shape = data.shape

        super().__init__(data=data, orig_shape=orig_shape)
        self._index = index
        self._path  = Path(path) if path is not None else None
        self._root  = Path(root) if root is not None else None

    @property
    def index(self) -> int:
        """Returns the index of the frame in the video."""
        return self._index

    @property
    def path(self) -> Path:
        """Returns the image file path."""
        return self._path

    @property
    def root(self) -> Path:
        """Returns the root directory for the image."""
        return self._root

    @property
    def frame_path(self) -> Path:
        """Returns the path for each frame of the video: <self.path>_<self.index>."""
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self.index}{SAVE_IMAGE_EXT}"
        else:
            return self._path

    @property
    def meta(self) -> dict:
        """Returns metadata about the image.

        Returns:
            A ``dict`` with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "path"      : self.frame_path,
            "video_path": self.path,
            "index"     : self.index,
            "orig_shape": self.orig_shape,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, Path) else None,
        }

    def load(self) -> np.ndarray:
        """Loads the image into memory.

        Returns:
            A ``numpy.ndarray`` of shape :math:`(H, W, C)` in :math:`[0, 255]`.
        """
        return self._data
