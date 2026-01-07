#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video base classes and mixins.

This module provides the base classes and mixins for videos.
"""

__all__ = [
    "Frame",
]

from typing import Any

import numpy as np

from mon.core.constants import SAVE_IMAGE_EXT
from mon.core.pathlib import Path
from ..base import Data, DataLoadMixin


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
class Frame(Data, DataLoadMixin):
    """A basic class for managing a video frame.
    
    Extend Data to handle a single video frame and provide properties and
    methods related to frame data.
    
    Attributes:
        _data (np.ndarray): An RGB or grayscale image, formatted as a
            numpy.ndarray with dimensions (H, W, C) and pixel values ranging
            from 0 to 255.
        _index (int): Index of the frame in the video.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data : np.ndarray,
        index: int,
        path : Path = None,
        root : Path = None,
    ):
        """Initialize a new instance.

        Args:
            data: An RGB or grayscale image, formatted as a numpy.ndarray with
                dimensions (H, W, C) and pixel values ranging from 0 to 255.
            index: Index of the frame in the video.
            path: Video file path. Defaults to None.
            root: Root directory of the video (of a dataset). Defaults to None.

        Raises:
            TypeError: If ``data`` is not a numpy.ndarray.
        """
        # Validate inputs
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}")
        
        # Initialize parent classes and assign attributes
        self._index = index
        super().__init__(data=data, path=path, root=root, persist=True)  # This will call the data setter
        
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the logical length of the container. For a frame, this is always 1."""
        return 1
    
    def __getitem__(self, idx: int = 0) -> np.ndarray:
        """Return the frame.

        Args:
            idx: Index to get the frame. Defaults to 0.
        """
        return self.data
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the underlying frame array."""
        return self._data
    
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the frame as (H, W, C)."""
        return self.data.shape
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the frame size as (H, W)."""
        return self.shape[0], self.shape[1]
    
    @property
    def index(self) -> int:
        """Return the frame index within the video."""
        return self._index

    @property
    def frame_path(self) -> Path:
        """Construct a path for the frame based on the video path and frame index.
        If no video path is provided, return the stored ``path``.
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
    
    # --- Data Loading ---
    def load(self, reload: bool = False) -> Any:
        """Dummy load method. Frames are expected to be provided directly and
        not loaded from disk.
        """
        pass
    
    def clear(self):
        """Clear the frame data from memory."""
        pass
    
