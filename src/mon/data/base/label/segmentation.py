#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements segmentation labels."""

from __future__ import annotations

__all__ = [
    "SegmentationLabel",
]

import uuid

import numpy as np

from mon import core
from mon.data.base.label import base

console = core.console


# region Segmentation

class SegmentationLabel(base.Label):
    """A semantic segmentation label in an image.
    
    See Also: :class:`Label`.
    
    Args:
        id_: The ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: The name of the image. Default: ``None``.
        path: The path to the image file. Default: ``None``.
        mask: The image with integer values encoding the semantic labels.
            Default: ``None``.
        load_on_create: If ``True``, the image will be loaded into memory when
            the object is created. Default: ``False``.
        keep_in_memory: If ``True``, the image will be loaded into memory and
            kept there. Default: ``False``.
    """

    to_rgb   : bool = True
    to_tensor: bool = False
    normalize: bool = False
    
    def __init__(
        self,
        id_           : int               = uuid.uuid4().int,
        name          : str        | None = None,
        path          : core.Path  | None = None,
        mask          : np.ndarray | None = None,
        load_on_create: bool              = False,
        keep_in_memory: bool              = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.id_            = id_
        self.image          = None
        self.keep_in_memory = keep_in_memory
        
        self.path = core.Path(path) if path is not None else None
        if self.path is None or not self.path.is_image_file():
            raise ValueError(
                f":param:`path` must be a valid path to an image file, "
                f"but got {path}."
            )
        
        if name is None:
            name = str(core.Path(path).name) if path.is_image_file() else f"{id_}"
        self.name = name

        if load_on_create and mask is None:
            mask = self.load()
        
        self.shape = core.get_image_shape(input=mask) if mask is not None else None
       
        if self.keep_in_memory:
            self.mask = mask
    
    def load(
        self,
        path          : core.Path | None = None,
        keep_in_memory: bool             = False,
    ) -> np.ndarray:
        """Load segmentation mask image into memory.
        
        Args:
            path: The path to the segmentation mask file. Default: ``None``.
            keep_in_memory: If ``True``, the image will be loaded into memory
                and kept there. Default: ``False``.
            
        Return:
            Return image of shape :math:`[H, W, C]`.
        """
        self.keep_in_memory = keep_in_memory
        
        if path is not None:
            path = core.Path(path)
            if path.is_image_file():
                self.path = path
        
        self.path = core.Path(path) if path is not None else None
        if self.path is None or not self.path.is_image_file():
            raise ValueError(
                f":param:`path` must be a valid path to an image file, "
                f"but got {path}."
            )
        
        mask = core.read_image(
            path      = self.path,
            to_rgb    = self.to_rgb,
            to_tensor = self.to_tensor,
            normalize = self.normalize,
        )
        self.shape = core.get_image_shape(input=mask) if (mask is not None) else self.shape
        
        if self.keep_in_memory:
            self.mask = mask
        return mask
        
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object."""
        return {
            "id"   : self.id_,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }
    
    @property
    def data(self) -> np.ndarray | None:
        """The label's data."""
        if self.mask is None:
            return self.load()
        else:
            return self.mask

# endregion
