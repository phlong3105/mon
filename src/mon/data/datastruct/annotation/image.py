#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of an image."""

from __future__ import annotations

__all__ = [
    "ImageAnnotation",
    "FrameAnnotation",
    "HeatmapAnnotation",
    "SegmentationAnnotation",
]

import uuid

import numpy as np

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Image Annotation

class ImageAnnotation(base.Annotation):
    """An image annotation for another image.
    
    Args:
        id_: An ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: A name of the image. Default: ``None``.
        path: A path to the image file. Default: ``None``.
        image: A ground-truth image to be loaded. Default: ``None``.
        load: If ``True``, the image will be loaded into memory when
            the object is created. Default: ``False``.
        cache: If ``True``, the image will be loaded into memory and
            kept there. Default: ``False``.

    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
     See Also: :class:`Annotation`.
    """
    
    def __init__(
        self,
        id_  : int               = uuid.uuid4().int,
        name : str        | None = None,
        path : core.Path  | None = None,
        image: np.ndarray | None = None,
        load : bool              = False,
        cache: bool              = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._id             = id_
        self._image          = None
        self._keep_in_memory = cache
        
        self._path = core.Path(path) if path is not None else None
        if (self.path is None or not self.path.is_image_file()) and image is None:
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {path}.")
        
        if name is None:
            name = str(core.Path(path).name) if path.is_image_file() else f"{id_}"
        self._name = name
        self._stem = core.Path(self._name).stem
        
        if load and image is None:
            image = self.load()
        
        self._shape = core.get_image_shape(input=image) if image is not None else None
        
        if self._keep_in_memory:
            self._image = image
    
    def load(
        self,
        path : core.Path | None = None,
        cache: bool 			= False,
    ) -> np.ndarray | None:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Default: ``None``.
            cache: If ``True``, the image will be loaded into memory
                and kept there. Default: ``False``.
            
        Return:
            An image of shape :math:`[H, W, C]`.
        """
        self._keep_in_memory = cache
        
        if path is not None:
            path = core.Path(path)
            if path.is_image_file():
                self._path = path
        if self.path is None or not self.path.is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {self.path}.")
        
        image = core.read_image(path=self.path, to_rgb=True, to_tensor=False, normalize=False)
        self._shape = core.get_image_shape(input=image) if (image is not None) else self._shape
        
        if self._keep_in_memory:
            self._image = image
        return image
    
    @property
    def path(self) -> core.Path:
        """The path to the image file."""
        return self._path
    
    @property
    def data(self) -> np.ndarray | None:
        if self._image is None:
            return self.load()
        else:
            return self._image
    
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "id"   : self._id,
            "name" : self._name,
            "stem" : self._stem,
            "path" : self.path,
            "shape": self._shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }


class FrameAnnotation(base.Annotation):
    """An image annotation in a video frame.
    
    Args:
        id_: An ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        index: An index of the frame. Default: ``None``.
        path: A path to the video file. Default: ``None``.
        frame: A ground-truth image to be loaded. Default: ``None``.
        cache: If ``True``, the image will be loaded into memory and
            kept there. Default: ``False``.

    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
     See Also: :class:`Annotation`.
    """
    
    def __init__(
        self,
        id_  : int               = uuid.uuid4().int,
        index: str        | None = None,
        path : core.Path  | None = None,
        frame: np.ndarray | None = None,
        cache: bool              = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._id             = id_
        self._index			 = index
        self._frame          = frame
        self._keep_in_memory = cache
        self._path 		     = core.Path(path) if path is not None else None
        self._name  		 = str(self.path.name) if self.path is not None else f"{id_}"
        self._stem  		 = core.Path(self._name).stem
        self._shape			 = core.get_image_shape(input=frame) if frame is not None else None
    
    @property
    def path(self) -> core.Path:
        """The path to the video file."""
        return self._path
    
    @property
    def data(self) -> np.ndarray | None:
        return self._frame
    
    @property
    def meta(self) -> dict:
        """Return a dictionary of metadata about the object. The dictionary
        includes ID, name, path, and shape of the image.
        """
        return {
            "id"   : self._id,
            "name" : self._name,
            "stem" : self._stem,
            "path" : self.path,
            "shape": self._shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

# endregion


# region Heatmap Annotation

class HeatmapAnnotation(base.Annotation):
    """A heatmap annotation for an image.
    
    See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
    
    Args:
        map: A 2D numpy array.
        range: An optional [min, max] range of the map's values. If None is
            provided, [0, 1] will be assumed if :param:`map` contains floating
            point values, and [0, 255] will be assumed if :param:`map` contains
            integer values.
    """
    
    @property
    def data(self) -> list | None:
        raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Segmentation Annotation

class SegmentationAnnotation(base.Annotation):
    """A segmentation annotation for an image.
    
    See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
    
    Args:
        id_: The ID of the image. This can be an integer or a string. This
            attribute is useful for batch processing where you want to keep the
            objects in the correct frame sequence.
        name: The name of the segmentation mask. Default: ``None``.
        path: The path to the image file. Default: ``None``.
        mask: The image with integer values encoding the semantic labels.
            Default: ``None``.
        load: If ``True``, the image will be loaded into memory when the object
            is created. Default: ``False``.
        cache: If ``True``, the image will be loaded into memory and kept there.
            Default: ``False``.
    """
    
    def __init__(
        self,
        id_  : int               = uuid.uuid4().int,
        name : str        | None = None,
        path : core.Path  | None = None,
        mask : np.ndarray | None = None,
        load : bool              = False,
        cache: bool              = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._id             = id_
        self._image          = None
        self._keep_in_memory = cache
        
        self._path = core.Path(path) if path is not None else None
        if self.path is None or not self.path.is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {path}.")
        
        if name is None:
            name = str(core.Path(path).name) if path.is_image_file() else f"{id_}"
        self._name = name
        self._stem = core.Path(self._name).stem
        
        if load and mask is None:
            mask = self.load()
        
        self._shape = core.get_image_shape(input=mask) if mask is not None else None
        
        if self._keep_in_memory:
            self._mask = mask
    
    def load(
        self,
        path : core.Path | None = None,
        cache: bool             = False,
    ) -> np.ndarray:
        """Load segmentation mask into memory.
        
        Args:
            path: The path to the segmentation mask file. Default: ``None``.
            cache: If ``True``, the image will be loaded into memory and kept
                there. Default: ``False``.
            
        Return:
            Return image of shape :math:`[H, W, C]`.
        """
        self._keep_in_memory = cache
        
        if path is not None:
            path = core.Path(path)
            if path.is_image_file():
                self._path = path
        
        self._path = core.Path(path) if path is not None else None
        if self.path is None or not self.path.is_image_file():
            raise ValueError(f":param:`path` must be a valid path to an image file, but got {path}.")
        
        mask = core.read_image(path=self.path, to_rgb=True, to_tensor=False, normalize=False)
        self._shape = core.get_image_shape(input=mask) if (mask is not None) else self._shape
        
        if self._keep_in_memory:
            self._mask = mask
        return mask
    
    @property
    def path(self) -> core.Path | None:
        return self._path
    
    @property
    def data(self) -> np.ndarray | None:
        if self._mask is None:
            return self.load()
        else:
            return self._mask
    
    @property
    def meta(self) -> dict:
        return {
            "id"   : self._id,
            "name" : self._name,
            "stem" : self._stem,
            "path" : self.path,
            "shape": self._shape,
            "hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

# endregion
