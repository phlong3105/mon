#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image labels."""

from __future__ import annotations

__all__ = [
    "ImageLabel",
]

import uuid

import numpy as np

from mon import core
from mon.data.base.label import base

console = core.console


# region Image

class ImageLabel(base.Label):
    """A ground-truth image label for an image.
    
    Args:
        id_: An ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: A name of the image. Default: ``None``.
        path: A path to the image file. Default: ``None``.
        image: A ground-truth image to be loaded. Default: ``None``.
        load_on_create: If ``True``, the image will be loaded into memory when
            the object is created. Default: ``False``.
        keep_in_memory: If ``True``, the image will be loaded into memory and
            kept there. Default: ``False``.

    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
     See Also: :class:`mon.data.base.label.base.Label`.
    """
    
    to_rgb   : bool = True
    to_tensor: bool = False
    normalize: bool = False
    
    def __init__(
        self,
        id_           : int               = uuid.uuid4().int,
        name          : str        | None = None,
        path          : core.Path  | None = None,
        image         : np.ndarray | None = None,
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

        if load_on_create and image is None:
            image = self.load()
        
        self.shape = core.get_image_shape(input=image) if image is not None else None
       
        if self.keep_in_memory:
            self.image = image
    
    def load(
        self,
        path          : core.Path | None = None,
        keep_in_memory: bool = False,
    ) -> np.ndarray | None:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Default: ``None``.
            keep_in_memory: If ``True``, the image will be loaded into memory
                and kept there. Default: ``False``.
            
        Return:
            An image of shape :math:`[H, W, C]`.
        """
        self.keep_in_memory = keep_in_memory
        
        if path is not None:
            path = core.Path(path)
            if path.is_image_file():
                self.path = path
        if self.path is None or not self.path.is_image_file():
            raise ValueError(
                f":param:`path` must be a valid path to an image file, "
                f"but got {self.path}."
            )
        
        image = core.read_image(
            path      = self.path,
            to_rgb    = self.to_rgb,
            to_tensor = self.to_tensor,
            normalize = self.normalize,
        )
        self.shape = core.get_image_shape(input=image) if (image is not None) else self.shape
        
        if self.keep_in_memory:
            self.image = image
        return image
        
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "id"   : self.id_,
            "name" : self.name,
            "path" : self.path,
            "shape": self.shape,
        }
    
    @property
    def data(self) -> np.ndarray | None:
        """The label's data."""
        if self.image is None:
            return self.load()
        else:
            return self.image
       
# endregion
