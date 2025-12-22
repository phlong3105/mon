#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video data classes and mixins.

This module provides the base classes and mixins for video data.
"""

__all__ = [
    "Frame",
]

import numpy as np

from mon.core.constants import SAVE_IMAGE_EXT
from mon.core.pathlib import Path
from ..base import Data


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---


# --- Compute Mixins ---


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
class Frame(Data):
    """A basic class for managing a video frame.

    Attributes:
        _data (np.ndarray): An RGB frame as a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
        _index (int): Index of the frame in the video.
        _path (Path): Video file path.
        _root (Path): Root directory of the video (of a dataset).
    """
    
    def __init__(
        self,
        data : np.ndarray,
        index: int,
        path : Path = None,
        root : Path = None,
    ):
        """Initialize the frame instance.

        Args:
            data: An RGB frame as a numpy.ndarray of shape (H, W, C) with pixel
                values in the range [0, 255].
            index: Index of the frame in the video.
            path: Video file path. Defaults to None.
            root: Root directory of the video (of a dataset). Defaults to None.

        Raises:
            TypeError: If data is not a numpy.ndarray.
        """
        # Validate inputs
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}")
         
        super().__init__(data=data)
        
        # Assign attributes
        self._index = index
        self._path  = Path(path) if path is not None else None
        self._root  = Path(root) if root is not None else None
    
    #---- Magic Methods ---
    def __len__(self) -> int:
        """Return the logical length of the container."""
        return 1
    
    def __getitem__(self, idx: int = 0) -> np.ndarray:
        """Return the frame image.

        Args:
            idx: Index to get the image. Defaults to 0.
        """
        return self.data
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the underlying frame array."""
        return self._data
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the frame shape as (H, W, C)."""
        return self.data.shape
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the frame size as (H, W)."""
        return self.data.shape[:2]
    
    @property
    def index(self) -> int:
        """Return the frame index within the video."""
        return self._index

    @property
    def path(self) -> Path:
        """Return the video file path associated with this frame."""
        return self._path

    @property
    def root(self) -> Path:
        """Return the root directory for the video dataset, if any."""
        return self._root

    @property
    def frame_path(self) -> Path:
        """Return a generated frame file path.

        Construct a path for the frame based on the video path and frame index.
        If no video path is provided, return the stored path.
        """
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self.index}{SAVE_IMAGE_EXT}"
        else:
            return self._path

    @property
    def meta(self) -> dict:
        """Return metadata dictionary for the frame.

        Include index, frame path, source video path, root, shape, and hash.
        """
        return {
            "index"     : self.index,
            "path"      : self.frame_path,
            "video_path": self.path,
            "root"      : self.root,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, Path) else None,
        }
