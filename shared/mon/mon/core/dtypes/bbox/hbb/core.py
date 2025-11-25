#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the data structure for horizontal bounding boxes (HBBs).
"""

__all__ = [
    "HBBs",
]

from typing import Union

import numpy as np
import torch

from mon.core.enum import BBoxFormat
from mon.core.pathlib import Path
from ...base import BaseTensorOrArray


class HBBs(BaseTensorOrArray):
    """HBBs object.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. If given,
            it must be of shape :math:`(N, 4+)` in ``XYXY`` format. Default: ``None``.
        orig_shape: Original shape of the image in :math:`(H, W)` format.
        path: Label file path. Default: ``None``.
        root: Root directory for the label file. Default: ``None``.
        fmt: One of the bounding box format from ``BBoxFormat``. Default: ``BBoxFormat.XYXY``.

    Notes:
        The bounding boxes are expected to in the following format:
            :math:`<x1, y1, x2, y2, conf, cls, id>`
        where:
            - :math:`<x1, y1, x2, y2>` are the bounding box coordinates in ``XYXY`` format.
            - :math:`<conf>` is the confidence score (optional).
            - :math:`<cls>` is the class ID (optional).
            - :math:`<id>` is the tracking ID (optional).
    """
    
    def __init__(
        self,
        data      : Union[torch.Tensor, np.ndarray] = None,
        orig_shape: tuple[int, int, int] = None,
        path      : Path = None,
        root      : Path = None,
        fmt       : BBoxFormat = BBoxFormat.XYXY,
    ):
        if all(d is None for d in [data, path]):
            raise ValueError("Either [data] or [path] must be provided to initialize the Image object.")
        if data is not None and orig_shape is None:
            raise ValueError(f"If ``data`` is provided, ``orig_shape`` must also be specified.")
        if data is not None and orig_shape is not None:
            from mon.core.dtypes.bbox.hbb import utils
            if not utils.is_xyxy(data, orig_shape):
                raise ValueError(f"``data`` must be in XYXY format, got {data.shape}.")

        super().__init__(data=data, orig_shape=orig_shape)
        self._path     = Path(path) if path is not None else None
        self._root     = Path(root) if root is not None else None
        self._orig_fmt = fmt
        self._cvt_fmt  = fmt
        self.fmt       = fmt

    @property
    def path(self) -> Path:
        """Returns the image file path."""
        return self._path

    @property
    def root(self) -> Path:
        """Returns the root directory for the image."""
        return self._root

    @property
    def fmt(self) -> BBoxFormat:
        """Returns the bounding box format."""
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: BBoxFormat):
        fmt = BBoxFormat.from_value(fmt)
        if fmt in BBoxFormat.conversion_codes():
            orig_fmt = BBoxFormat.from_value(fmt.value.split("_to_")[0])
        else:
            orig_fmt = fmt

        # We default to XYXY format for HBBs
        fmt = BBoxFormat.XYXY
        if orig_fmt != fmt:
            cvt_fmt = BBoxFormat.from_value(f"{orig_fmt.value}_to_{fmt.value}")
        else:
            cvt_fmt = fmt

        self._fmt      = fmt
        self._orig_fmt = orig_fmt
        self._cvt_fmt  = cvt_fmt

    @property
    def xyxy(self) -> Union[torch.Tensor, np.ndarray]:
        """Returns the bounding boxes as a ``torch.Tensor`` or ``numpy.ndarray``
        of shape :math:`(N, 4)` in ``XYXY`` format (default).
        """
        return self.data[:, :4]

    @property
    def conf(self) -> Union[torch.Tensor, np.ndarray]:
        """Return the confidence scores for each detection box as a ``torch.Tensor``
        or ``numpy.ndarray`` of shape :math:`(N, 1)`.
        """
        if self.data.shape[1] > 4:
            return self.data[:, 4:5]
        return None

    @property
    def cls(self) -> Union[torch.Tensor, np.ndarray]:
        """Return the class ID tensor representing category predictions for each
        bounding box as a ``torch.Tensor`` or ``numpy.ndarray`` of shape :math:`(N, 1)`.
        """
        if self.data.shape[1] > 5:
            return self.data[:, 5:6]
        return None

    @property
    def id(self) -> Union[torch.Tensor, np.ndarray]:
        """Return the tracking IDs for each detection box (if available) as
        ``torch.Tensor`` or ``numpy.ndarray`` of shape :math:`(N, 1)`.

        Notes:
            - This property is only available when tracking is enabled.
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        if self.data.shape[1] > 6:
            return self.data[:, 6:7]
        return None

    @property
    def xywh(self) -> Union[torch.Tensor, np.ndarray]:
        """Returns the bounding boxes as a ``torch.Tensor`` or ``numpy.ndarray``
        of shape :math:`(N, 4)` in ``XYWH`` format.
        """
        from .processing import xyxy_to_xywh
        return xyxy_to_xywh(self.data[:, :4], self.orig_shape)

    @property
    def xyxyn(self) -> Union[torch.Tensor, np.ndarray]:
        """Returns the bounding boxes as a ``torch.Tensor`` or ``numpy.ndarray``
        of shape :math:`(N, 4)` in ``XYXYN`` format.
        """
        from .processing import normalize
        return normalize(self.data[:, :4], self.orig_shape)

    @property
    def cxcywhn(self) -> Union[torch.Tensor, np.ndarray]:
        """Returns the bounding boxes as a ``torch.Tensor`` or ``numpy.ndarray``
        of shape :math:`(N, 4)` in ``CXCYWH`` format.
        """
        from .processing import xyxy_to_cxcywhn
        return xyxy_to_cxcywhn(self.data[:, :4], self.orig_shape)

    def load(self, reload: bool = False) -> np.ndarray:
        """Loads the image into memory.

        Args:
            reload: If ``True``, reload the image even if already cached.
                Default: ``False``.

        Returns:
            A ``numpy.ndarray`` of shape :math:`(H, W, C)` in :math:`[0, 255]`.
        """
        # Return the bbox if it is already loaded and not reloading
        if not reload and self._data is not None:
            return self._data

        # Load the image
        from .io import load
        bbox = load(path=self.path, fmt=self._cvt_fmt, imgsz=self.orig_shape)

        # Cache
        self._data = bbox
        return bbox
