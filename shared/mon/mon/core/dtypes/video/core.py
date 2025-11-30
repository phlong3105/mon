#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for video data type.

This module provides classes for handling video and video frames, including
loading, caching, and accessing frame properties. It supports frames stored
as in-memory arrays or tensors, with metadata access.
"""

__all__ = [
    "Frame",
]

import numpy as np

from mon.core.constants import SAVE_IMAGE_EXT
from mon.core.pathlib import Path
from ..base import Data


class Frame(Data):
    """A base class for a single video frame.
    
    This class extends Data to handle a single frame from a video, which is
    provided as an in-memory array/tensor. It provides properties to access
    frame metadata such as index, path, and shape.
    """
    
    def __init__(
        self,
        data : np.ndarray,
        index: int,
        path : Path = None,
        root : Path = None,
    ):
        """Initializes the Frame instance.
        
        Args:
            data (numpy.ndarray): An RGB image as a numpy.ndarray of shape
                (H, W, C) with pixel values in the range [0, 255].
            index (int): The index of the frame in the video.
            path (Path, optional): Video file path. Defaults to None.
            root (Path, optional): Root directory of the video (of a dataset).
                Defaults to None.
        """
        super().__init__()
        # Validate inputs
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}")
            
        # Assign attributes
        self._data  = data
        self._index = index
        self._path  = Path(path) if path is not None else None
        self._root  = Path(root) if root is not None else None
    
    #---- Magic Methods -----
    def __len__(self) -> int:
        """Returns the length of 1 (i.e., a single frame)."""
        return 1
    
    def __getitem__(self, idx: int = 0) -> np.ndarray:
        """Returns the frame itself.
        
        Args:
            idx (int): Index to get the image. Defaults to 0.
        
        Returns:
            numpy.ndarray: The image itself.
        """
        return self.data
    
    # ----- Properties -----
    @property
    def data(self) -> np.ndarray:
        """Getter for the frame.
        
        Returns:
            numpy.ndarray: The frame as a numpy.ndarray of shape (H, W, C) with
                pixel values in the range [0, 255].
        """
        return self._data
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Getter for the original shape of the frame as (H, W, C).
        
        Returns:
            tuple[int, int, int]: The original shape of the frame.
        """
        return self.data.shape
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Getter for the frame size as (H, W).
        
        Returns:
            tuple[int, int]: The frame size.
        """
        return self.data.shape[:2]
    
    @property
    def index(self) -> int:
        """Getter for the frame index in the video.
        
        Returns:
            int: The frame index.
        """
        return self._index

    @property
    def path(self) -> Path:
        """Getter for the video file path.
        
        Returns:
            Path: The video file path.
        """
        return self._path

    @property
    def root(self) -> Path:
        """Getter for the root directory of the video (i.e., dataset root).
        
        Returns:
            Path: The root directory of the videos.
        """
        return self._root

    @property
    def frame_path(self) -> Path:
        """Getter for the frame file path.
        
        This constructs a path for the frame based on the video path and frame
        index. If the video path is not provided, it returns a default path.
        
        Returns:
            Path: The frame file path.
        """
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self.index}{SAVE_IMAGE_EXT}"
        else:
            return self._path

    @property
    def meta(self) -> dict:
        """Getter for metadata about the frame.
        
        Returns:
            dict: Metadata about the frame.
        """
        return {
            "index"     : self.index,
            "path"      : self.frame_path,
            "video_path": self.path,
            "root"      : self.root,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, Path) else None,
        }
    
    # ----- Initialize -----
    def load(self, reload: bool = False) -> np.ndarray:
        """No need to load data from disk, just return the underlying data.
        
        Args:
            reload (bool): Ignored for this base class. Defaults to False.
            
        Returns:
            numpy.ndarray: The underlying frame.
        """
        return self.data
    
    # ----- Device Management Methods -----
    def cpu(self):
        """Do nothing.
        
        Raises:
            NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError("Not implemented yet.")
    
    def cuda(self):
        """Do nothing.
        
        Raises:
            NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError("Not implemented yet.")
    
    def numpy(self):
        """Do nothing.
        
        Raises:
            NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError("Not implemented yet.")
    
    def to(self, *args, **kwargs):
        """"Do nothing.
        
        Raises:
            NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError("Not implemented yet.")
