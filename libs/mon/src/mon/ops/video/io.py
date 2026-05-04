#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video I/O Operation.

This module provides input and output operations for videos.
"""

from __future__ import annotations

__all__ = [
    "VideoWriter",
    "VideoWriterCV",
]

from abc import ABC, abstractmethod
from typing import override

import cv2
from numpy import ndarray
from torch import Tensor

from mon.core import Path, Size, TensorOrArray
from mon.ops.image import to_image_array


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================

# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

class VideoWriter(ABC):
    """Abstract class for video writers."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        path: Path,
        imgsz: Size = (480, 640),
        frame_rate: float = 24,
        verbose: bool = False,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            path (Path): Video output file.
            imgsz (Size, optional): Output video resolution as (H, W).
                Defaults to (480, 640).
            frame_rate (float, optional): Output video frame rate. Defaults to 24.
            verbose (bool, optional): Verbosity mode. Defaults to False.
        """
        # Validate inputs
        if isinstance(path, (Path, str)):
            raise TypeError(
                f"Expected 'path' to point to a video file, "
                f"but got: {type(path).__name__}."
            )
        path = Path(path).normalize()
        if not path.video_file():
            raise ValueError(f"Expected 'path' to point to a video file, but got: '{path.as_posix()}'.")

        if frame_rate <= 0:
            raise ValueError(f"Expected 'frame_rate' to be positive, but got: {frame_rate}.")

        # Assign attributes
        self.verbose = verbose
        self.path = path
        self.imgsz = Size.from_value(imgsz)
        self.frame_rate = frame_rate
        self.cur_idx = 0
        self.init()

    @abstractmethod
    def init(self):
        """Create and configure backend-specific writer resources."""
        pass

    def __del__(self):
        """Finalizer called when the object is about to be destroyed."""
        self.close()

    @abstractmethod
    def close(self):
        """Release system resources."""
        pass

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the number of written frames."""
        return self.cur_idx

    # --- Callable & Context Manager ---
    def __call__(
        self,
        frame: TensorOrArray,
        path: Path | None = None,
        *args, **kwargs
    ):
        """Write a ``frame`` to the video output.

        Args:
            frame (TensorOrArray): Video frame, formatted as an array of
                shape (H, W, C) and values ranging from 0 to 255; or as a
                tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.
            path (Path | None, optional): Optional path to also save the ``frame``
                as an image. Defaults to None.

        Raises:
            TypeError: If ``frame`` is not an array or tensor.
        """
        if not isinstance(frame, (ndarray, Tensor)):
            raise TypeError(
                f"Expected 'frame' to be a 'np.ndarray' or 'torch.Tensor', "
                f"but got: {type(frame).__name__}."
            )
        self.write(frame)

    def __enter__(self):
        """Setup for 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Teardown for 'with' statement."""
        self.close()

    # --- Write ---
    @abstractmethod
    def write(self, frame: TensorOrArray):
        """Internal method for backend-specific writing logic.

        Args:
            frame (TensorOrArray): Video frame, formatted as an array of shape
                (H, W, C) and values ranging from 0 to 255; or as a tensor of
                shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        pass


class VideoWriterCV(VideoWriter):
    """Video writer using OpenCV."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        path: Path,
        imgsz: Size = (480, 640),
        frame_rate: float = 24,
        fourcc: str = "mp4v",
        verbose: bool = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            Args:
            path (Path): Video output file.
            imgsz (Size, optional): Output video resolution as (H, W).
                Defaults to (480, 640).
            frame_rate (float, optional): Output video frame rate. Defaults to 24.
            fourcc (str, optional): FourCC code for the video codec. Defaults to "mp4v".
            verbose (bool, optional): Verbosity mode. Defaults to False.
        """
        # Validate inputs
        if not isinstance(fourcc, str):
            raise TypeError(f"Expected 'fourcc' to be a str, but got: {type(fourcc).__name__}.")

        # Assign attributes
        self.fourcc = fourcc
        self.video_writer = None

        # Continue the initialization chain
        super().__init__(
            path=path,
            imgsz=imgsz,
            frame_rate=frame_rate,
            verbose=verbose,
            *args, **kwargs
        )

    @override
    def init(self):
        """Initialize the OpenCV video writer.

        Raises:
            RuntimeError: If the video writer cannot be opened.
        """
        # 1. Resolve the output video file path
        if self.path.is_dir():
            video_file = self.path / f"result.mp4"
        else:
            video_file = self.path.parent / f"{self.path.stem}.mp4"

        video_file.parent.mkdir(parents=True, exist_ok=True)

        # 2. Initialize the video writer
        h, w = self.imgsz.hw
        self.video_writer = cv2.VideoWriter(
            filename=str(video_file),
            fourcc=cv2.VideoWriter_fourcc(*self.fourcc),
            fps=float(self.frame_rate),
            frameSize=(w, h),  # OpenCV uses (W, H) format
            isColor=True
        )

        if not self.video_writer.isOpened():
            raise RuntimeError(f"Could not open 'VideoWriter' at: {video_file}.")

    @override
    def close(self):
        """Release system resources."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    # --- Write ---
    def write(self, frame: TensorOrArray):
        """Internal method for backend-specific writing logic.

        Args:
            frame (TensorOrArray): Video frame, formatted as an array of shape
                (H, W, C) and values ranging from 0 to 255; or as a tensor of
                shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        # 1. Convert to an RGB array
        if isinstance(frame, Tensor):
            frame = to_image_array(frame)

        # 2. Resize if the frame size doesn't match initialization
        h, w = self.imgsz.hw
        fh, fw = frame.shape[:2]
        if (fh, fw) != (h, w):
            frame = cv2.resize(frame, (w, h))

        # 3. Convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 4. Write the frame
        self.video_writer.write(frame_bgr)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
