#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of an image."""

from __future__ import annotations

__all__ = [
    "ImageAnnotation",
    "FrameAnnotation",
    "SegmentationAnnotation",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Image

class ImageAnnotation(base.Annotation):
    """An image annotation for another image.
    
    Args:
        path: A path to the image file.
        flags: A flag to read the image. One of:
            - cv2.IMREAD_UNCHANGED           = -1,
            - cv2.IMREAD_GRAYSCALE           = 0,
            - cv2.IMREAD_COLOR               = 1,
            - cv2.IMREAD_ANYDEPTH            = 2,
            - cv2.IMREAD_ANYCOLOR            = 4,
            - cv2.IMREAD_LOAD_GDAL           = 8,
            - cv2.IMREAD_REDUCED_GRAYSCALE_2 = 16,
            - cv2.IMREAD_REDUCED_COLOR_2     = 17,
            - cv2.IMREAD_REDUCED_GRAYSCALE_4 = 32,
            - cv2.IMREAD_REDUCED_COLOR_4     = 33,
            - cv2.IMREAD_REDUCED_GRAYSCALE_8 = 64,
            - cv2.IMREAD_REDUCED_COLOR_8     = 65,
            - cv2.IMREAD_IGNORE_ORIENTATION  = 128
            Default: ``cv2.IMREAD_COLOR``.
        
    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
    """
    
    def __init__(
        self,
        path : core.Path | str,
        flags: int = cv2.IMREAD_COLOR,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.path   = path
        self.flags  = flags
        self.image  = None
        self._shape = None
    
    @property
    def path(self) -> core.Path:
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f"`path` must be a valid path to an image file, but got {path}.")
        self._path  = core.Path(path)
        self._shape = core.read_image_shape(path=self._path)
    
    @property
    def name(self) -> str:
        return str(self.path.name)
    
    @property
    def stem(self) -> str:
        return str(self.path.stem)
    
    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape
    
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
    
    def load(
        self,
        path : core.Path | str | None = None,
        flags: int | None = None,
        cache: bool       = False,
    ) -> np.ndarray:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Default: ``None``.
            flags: A flag to read the image. Default: ``None``.
            cache: If ``True``, the image will be loaded into memory and kept
                there. Default: ``False``.
            
        Return:
            An image of shape `[H, W, C]`.
        """
        if self.image is not None:
            return self.image
        
        self.path  = path  if path  else self.path
        self.flags = flags if flags else self.flags
        image = core.read_image(
            path      = self.path,
            flags     = self.flags,
            to_tensor = False,
            normalize = False
        )
        self.image = image if cache else None
        return image
    
    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        keepdim  : bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
            keepdim: If ``True``, keep the dimensions of the input data. Default: ``False``.
            normalize: If ``True``, normalize the input data. Default: ``True``.
        """
        return core.to_image_tensor(image=data, keepdim=keepdim, normalize=normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collate function used to fused input items together when using
		:obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A :obj:`list` of images.
		"""
        return core.to_4d_image(batch)
    

class FrameAnnotation(base.Annotation):
    """An image annotation of a video frame.
    
    Args:
        index: The index of the frame in the video.
        path: A path to the video file. Default: ``None``.
        frame: A ground-truth image to be loaded. Default: ``None``.
        
    References:
        `<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
        
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
        self.path  = path
        self.shape = core.get_image_shape(image=frame)
    
    @property
    def path(self) -> core.Path:
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        if path is None or not core.Path(path).is_video_file():
            raise ValueError(f"`path` must be a valid path to a video file, but got {path}.")
        self._path = core.Path(path)
    
    @property
    def name(self) -> str:
        return str(self.path.name) if self.path else f"{self.index}"
    
    @property
    def stem(self) -> str:
        return str(self.path.stem) if self.path else f"{self.index}"
    
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
    
    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        keepdim  : bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
            keepdim: If ``True``, keep the dimensions of the input data. Default: ``False``.
            normalize: If ``True``, normalize the input data. Default: ``True``.
        """
        return core.to_image_tensor(image=data, keepdim=keepdim, normalize=normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collate function used to fused input items together when using
		:obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A :obj:`list` of images.
		"""
        return core.to_4d_image(batch)
    
# endregion


# region Segmentation

class SegmentationAnnotation(base.Annotation):
    """A segmentation annotation (mask) for an image.
    
    Args:
        path: The path to the image file.
        flags: A flag to read the image. One of:
            - cv2.IMREAD_UNCHANGED           = -1,
            - cv2.IMREAD_GRAYSCALE           = 0,
            - cv2.IMREAD_COLOR               = 1,
            - cv2.IMREAD_ANYDEPTH            = 2,
            - cv2.IMREAD_ANYCOLOR            = 4,
            - cv2.IMREAD_LOAD_GDAL           = 8,
            - cv2.IMREAD_REDUCED_GRAYSCALE_2 = 16,
            - cv2.IMREAD_REDUCED_COLOR_2     = 17,
            - cv2.IMREAD_REDUCED_GRAYSCALE_4 = 32,
            - cv2.IMREAD_REDUCED_COLOR_4     = 33,
            - cv2.IMREAD_REDUCED_GRAYSCALE_8 = 64,
            - cv2.IMREAD_REDUCED_COLOR_8     = 65,
            - cv2.IMREAD_IGNORE_ORIENTATION  = 128
            Default: ``cv2.IMREAD_COLOR``.
            
    """
    
    def __init__(
        self,
        path : core.Path | str,
        flags: int = cv2.IMREAD_COLOR,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.path   = path
        self.flags  = flags
        self.mask   = None
        self._shape = None
    
    @property
    def path(self) -> core.Path:
        return self._path
    
    @path.setter
    def path(self, path: core.Path | str | None):
        if path is None or not core.Path(path).is_image_file():
            raise ValueError(f"`path` must be a valid path to an image file, but got {path}.")
        self._path  = core.Path(path)
        self._shape = core.read_image_shape(path=self._path)
    
    @property
    def name(self) -> str:
        return str(self.path.name)
    
    @property
    def stem(self) -> str:
        return str(self.path.stem)
    
    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape
    
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
    
    def load(
        self,
        path : core.Path | str | None = None,
        flags: int | None = None,
        cache: bool       = False,
    ) -> np.ndarray | None:
        """Loads image into memory.
        
        Args:
            path: The path to the image file. Default: ``None``.
            flags: A flag to read the image. Default: ``None``.
            cache: If ``True``, the image will be loaded into memory and kept
                there. Default: ``False``.
            
        Return:
            An image of shape `[H, W, C]`.
        """
        if self.mask is not None:
            return self.mask
        
        self.path  = path  if path  else self.path
        self.flags = flags if flags else self.flags
        mask = core.read_image(
            path      = self.path,
            flags     = self.flags,
            to_tensor = False,
            normalize = False
        )
        self.mask = mask if cache else None
        return mask
    
    @staticmethod
    def to_tensor(
        data     : torch.Tensor | np.ndarray,
        keepdim  : bool = False,
        normalize: bool = True
    ) -> torch.Tensor:
        """Converts the input data to a :obj:`torch.Tensor`.
        
        Args:
            data: The input data.
            keepdim: If ``True``, keep the dimensions of the input data. Default: ``False``.
            normalize: If ``True``, normalize the input data. Default: ``True``.
        """
        return core.to_image_tensor(image=data, keepdim=keepdim, normalize=normalize)
    
    @staticmethod
    def collate_fn(batch: list[torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray | None:
        """Collate function used to fused input items together when using
		:obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A :obj:`list` of images.
		"""
        return core.to_4d_image(batch)

# endregion
