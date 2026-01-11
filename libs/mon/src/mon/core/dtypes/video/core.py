#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video data structures.

This module provides the base classes and mixins for video.
"""

from __future__ import annotations

__all__ = [
    "Frame",
]

from typing import Any, Optional

import numpy as np

from mon.core.constants import EXT
from mon.core.dtypes.base import PersistentData
from mon.core.pathlib import Path


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class Frame(PersistentData):
    """A basic class for managing a video frame.

    Extend Data to handle a single video frame and provide properties and
    methods related to frame data.

    Attributes:
        _data (np.ndarray): An RGB or grayscale image, formatted as a
            numpy.ndarray of shape (H, W, C) and pixel values ranging from 0 to
            255.
        _index (int): Index of the frame in the video.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data : np.ndarray,
        index: int,
        path : Path | str | None = None,
        root : Path | str | None = None,
    ):
        """Initialize a new instance.

        Args:
            data: An RGB or grayscale image, formatted as a numpy.ndarray of
                shape (H, W, C) and pixel values ranging from 0 to 255.
            index: Index of the frame in the video.
            path: Video file path. Defaults to None.
            root: Root directory of the video (of a dataset). Defaults to None.

        Raises:
            TypeError: If ``data`` is not a numpy.ndarray.
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected 'data' to be a numpy.ndarray, but got {type(data)}.")
        
        # Set internal attributes
        self._index = index
        
        # Continue the initialization chain
        super().__init__(data=data, path=path, root=root, persist=True)
        
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the logical length of the container.
        
        For a frame, this is always 1.
        """
        return 1
    
    def __getitem__(self, index: int = 0) -> np.ndarray:
        """Return the frame.

        Args:
            index: Index to get the frame. Defaults to 0.
        """
        return self.data
    
    # --- Properties ---
    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the frame shape as (H, W, C)."""
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
    def frame_path(self) -> Optional[Path]:
        """Construct a path for the frame based on the video path and frame
        index.
        
        If no video path is provided, return the stored ``path``.
        """
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self._index}{EXT.IMAGE}"
        else:
            return self.path

    @property
    def meta(self) -> dict:
        """Return metadata describing the data."""
        return {
            "index"     : self._index,
            "path"      : self.frame_path,
            "video_path": self.path,
            "root"      : self.root,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, Path) else None,
        }
    
    # --- Data Loading ---
    def load(self, reload: bool = False) -> Any:
        """Load the data.
        
        Frames are expected to be provided directly and not loaded from the
        disk.
        """
        pass

# endregion
