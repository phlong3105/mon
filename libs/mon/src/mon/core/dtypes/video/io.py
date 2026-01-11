#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video I/O operations.

This module provides input and output operations for video.
"""

from __future__ import annotations

__all__ = [
    "VideoWriter",
    "VideoWriterCV",
]

import abc

import cv2
import numpy as np
import torch

from mon.core.dtypes import image as I
from mon.core.pathlib import Path


# ==============================================================================
# region DISCOVERY
# ==============================================================================


# endregion


# ==============================================================================
# region CONNECTION
# ==============================================================================


# endregion


# ==============================================================================
# region INPUT
# ==============================================================================


# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

class VideoWriter(abc.ABC):
    """An abstract class for video writers.

    Define the interface for writing frames to video files; subclasses must
    implement initialization, closing, and frame writing.

    Attributes:
        _dst (Path): Destination path or directory for the video output.
        _imgsz (tuple[int, int]): Output video size as (H, W).
        _frame_rate (float): Output video frame rate.
        _cur_idx (int): Current number of written frames.
        verbose (bool): Enable verbosity.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        dst       : Path | str,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float           = 30,
        verbose   : bool            = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            dst: Destination path or directory for the video output.
            imgsz: Output video size as (H, W).
            frame_rate: Output video frame rate.
            verbose: Enable verbosity.
        """
        if not isinstance(dst, (str, Path)):
            raise TypeError(f"Expected 'dst' to be a str or Path, but got {type(dst).__name__}.")
        if frame_rate <= 0:
            raise ValueError(f"Expected 'frame_rate' to be positive, but got {frame_rate}.")
            
        self.verbose     = verbose
        self._cur_idx    = 0
        self._dst        = Path(dst).normalize()
        self._imgsz      = I.imgsz(imgsz)
        self._frame_rate = frame_rate
        self._init()
     
    def __del__(self):
        """Close resources held by the writer."""
        self.close()
    
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the number of written frames."""
        return self._cur_idx
    
    # --- Callable & Context Manager ---
    def __call__(
        self,
        frame: np.ndarray | torch.Tensor,
        path : Path | str | None = None,
        *args, **kwargs
    ):
        """Write a frame to the video output.

        Args:
            frame: A video frame, formatted as a numpy.ndarray of shape
                (H, W, C) and pixel values ranging from 0 to 255; or as a
                torch.Tensor of shape (B, C, H, W) with pixel values
                ranging from 0.0 to 1.0.
            path: Optional path to also save the frame as an image.
        """
        if not isinstance(frame, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Expected 'frame' to be a 'np.ndarray' or 'torch.Tensor', "
                f"but got {type(frame).__name__}."
            )
        self._write(frame)
        self._cur_idx += 1
    
    def __enter__(self):
        """Setup for 'with' statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Teardown for 'with' statement."""
        self.close()
    
    # --- Properties ---
    @property
    def cur_idx(self) -> int:
        """Return the current written frame index."""
        return self._cur_idx
    
    @property
    def dst(self) -> Path:
        """Return the destination path for the output video."""
        return self._dst
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the configured frame size as (H, W)."""
        return self._imgsz
    
    @property
    def frame_rate(self) -> float:
        """Return the configured frame rate."""
        return self._frame_rate
    
    # --- Initialize ---
    @abc.abstractmethod
    def _init(self):
        """Create and configure backend-specific writer resources."""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Release system resources."""
        pass
    
    # --- Write ---
    @abc.abstractmethod
    def _write(self, frame: np.ndarray | torch.Tensor):
        """Internal method for backend-specific writing logic.
        
        Args:
            frame: A video frame, formatted as a numpy.ndarray of shape
                (H, W, C) and pixel values ranging from 0 to 255; or as a
                torch.Tensor of shape (B, C, H, W) with pixel values
                ranging from 0.0 to 1.0.
        """
        pass
        

class VideoWriterCV(VideoWriter):
    """A video writer using OpenCV.

    Extend VideoWriter to implement video writing using OpenCV's VideoWriter
    class.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        dst       : Path | str,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float           = 30,
        fourcc    : str             = "mp4v",
        verbose   : bool            = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            dst: Destination path or directory for the video output.
            imgsz: Output video size as a tuple of (H, W).
            frame_rate: Output video frame rate.
            fourcc: FourCC code for the video codec.
            verbose: Enable verbosity.
        """
        if not isinstance(fourcc, str):
            raise TypeError(f"Expected 'fourcc' to be a str, but got {type(fourcc).__name__}.")

        self._fourcc       = fourcc
        self._video_writer = None
        
        # Continue the initialization chain
        super().__init__(
            dst        = dst,
            imgsz      = imgsz,
            frame_rate = frame_rate,
            verbose    = verbose,
            *args, **kwargs
        )
    
    def __del__(self):
        """Close video writer."""
        if self._video_writer:
            self._video_writer.release()
    
    # ---- Initialize ---
    def _init(self):
        """Initialize the OpenCV video writer.
        
        Raises:
            RuntimeError: If the video writer cannot be opened.
        """
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
            
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        # OpenCV uses (Width, Height)
        w, h = self.imgsz[1], self.imgsz[0]
        
        self._video_writer = cv2.VideoWriter(
            filename  = str(video_file),
            fourcc    = cv2.VideoWriter_fourcc(*self._fourcc),
            fps       = float(self.frame_rate),
            frameSize = (w, h),
            isColor   = True
        )
        
        if not self._video_writer.isOpened():
            raise RuntimeError(f"Could not open 'VideoWriter' at: {video_file}.")
    
    def close(self):
        """Close the video writer."""
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
    
    # --- Write ---
    def _write(self, frame: np.ndarray | torch.Tensor):
        """Internal method for backend-specific writing logic.
        
        Args:
            frame: A video frame, formatted as a numpy.ndarray of shape
                (H, W, C) and pixel values ranging from 0 to 255; or as a
                torch.Tensor of shape (B, C, H, W) with pixel values
                ranging from 0.0 to 1.0.
        """
        # Convert to NumPy uint8 RGB [H, W, 3]
        if isinstance(frame, torch.Tensor):
            frame = I.to_array(frame)
        
        # Safety Resize: If frame size doesn't match initialization
        fh, fw = frame.shape[:2]
        if (fh, fw) != self.imgsz:
            frame = cv2.resize(frame, (self.imgsz[1], self.imgsz[0]))
        
        # Color Space: OpenCV expects BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self._video_writer.write(frame_bgr)

# endregion
