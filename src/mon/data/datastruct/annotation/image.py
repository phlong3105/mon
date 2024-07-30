#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of an image."""

from __future__ import annotations

__all__ = [
    "ImageAnnotation",
    "FrameAnnotation",
    "SegmentationAnnotation",
]

import numpy as np

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Image

class ImageAnnotation(base.Annotation):
    """An image annotation for another image.
    
    Args:
        path: A path to the image file.
        
    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
    See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
    """
    
    def __init__(self, path: core.Path | str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {self.path}.")
        self.path  = core.Path(path)
        self.name  = str(self.path.name)
        self.stem  = str(self.path.stem)
        self.image = None
        self.shape = core.read_image_shape(path=self.path)
    
    def load(
        self,
        path : core.Path | str | None = None,
        cache: bool = False,
    ) -> np.ndarray | None:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Default: ``None``.
            cache: If ``True``, the image will be loaded into memory
                and kept there. Default: ``False``.
            
        Return:
            An image of shape :math:`[H, W, C]`.
        """
        if path is not None and core.Path(path).is_image_file():
            self.path = core.Path(path)
        if self.path is None or not self.path.is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {self.path}.")
        
        image      = core.read_image(path=self.path, to_rgb=True, to_tensor=False, normalize=False)
        self.shape = core.get_image_shape(input=image) if (image is not None) else self.shape
        
        if cache:
            self.image = image
        return image
    
    @property
    def data(self) -> np.ndarray | None:
        if self.image is None:
            return self.load(cache=False)
        else:
            return self.image
    
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }


class FrameAnnotation(base.Annotation):
    """An image annotation of a video frame.
    
    Args:
        index: The index of the frame in the video.
        path: A path to the video file. Default: ``None``.
        frame: A ground-truth image to be loaded. Default: ``None``.
        
    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
    See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
    """
    
    def __init__(
        self,
        index: int,
        frame: np.ndarray,
        path : core.Path | str | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.frame = frame
        self.path  = core.Path(path)     if path      is not None else None
        self.name  = str(self.path.name) if self.path is not None else f"{index}"
        self.stem  = str(self.path.stem)
        self.shape = core.get_image_shape(input=frame)
    
    @property
    def data(self) -> np.ndarray | None:
        return self.frame
    
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "index": self.index,
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

# endregion


# region Segmentation

class SegmentationAnnotation(base.Annotation):
    """A segmentation annotation (mask) for an image.
    
    Args:
        path: The path to the image file.
    
    See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
    """
    
    def __init__(
        self,
        path: core.Path | str,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {self.path}.")
        self.path  = core.Path(path)
        self.name  = str(self.path.name)
        self.stem  = str(self.path.stem)
        self.mask  = None
        self.shape = core.read_image_shape(path=self.path)
    
    def load(
        self,
        path : core.Path | str | None = None,
        cache: bool = False,
    ) -> np.ndarray | None:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Default: ``None``.
            cache: If ``True``, the image will be loaded into memory
                and kept there. Default: ``False``.
            
        Return:
            An image of shape :math:`[H, W, C]`.
        """
        if path is not None and core.Path(path).is_image_file():
            self.path = core.Path(path)
        if self.path is None or not self.path.is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {self.path}.")
        
        mask       = core.read_image(path=self.path, to_rgb=True, to_tensor=False, normalize=False)
        self.shape = core.get_image_shape(input=mask) if (mask is not None) else self.shape
        
        if cache:
            self.mask = mask
        return mask
    
    @property
    def data(self) -> np.ndarray | None:
        if self.mask is None:
            return self.load()
        else:
            return self.mask
    
    @property
    def meta(self) -> dict:
        return {
            "name" : self.name,
            "stem" : self.stem,
            "path" : self.path,
            "shape": self.shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

# endregion
