#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video data I/O operations.

This module provides functions for input and output operations for video data.
"""

__all__ = [
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "load_video_ffmpeg",
    "write_video_ffmpeg",
]

import abc
from typing import Union

import cv2
import ffmpeg
import numpy as np
import torch

from mon.core.pathlib import Path
from .. import image as I


# --- Reading ---
def load_video_ffmpeg(process, height: int, width: int) -> np.ndarray:
    """Read a frame from an ffmpeg process.

    Args:
        process: Subprocess managing ffmpeg.
        height: Height of the output frame.
        width: Width of the output frame.

    Raises:
        ValueError: If the number of bytes read does not match the expected size.
    """
    # RGB24: 3 bytes per pixel
    img_size = height * width * 3
    in_bytes = process.stdout.read(img_size)
    if len(in_bytes) == 0:
        image = None
    else:
        if len(in_bytes) != img_size:
            raise ValueError(f"``in_bytes`` length [{len(in_bytes)}] != expected size [{img_size}].")
        image = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return image


# --- Writing ---
def write_video_ffmpeg(process, frame: Union[torch.Tensor, np.ndarray]):
    """Write a frame to an ffmpeg process.

    Args:
        process: Subprocess managing ffmpeg.
        frame: Frame as an RGB numpy array.

    Raises:
        ValueError: If frame is not a numpy.ndarray.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"``frame`` must be a numpy.ndarray, got {type(frame).__name__}.")
    process.stdin.write(
        frame
        .astype("uint8")
        .tobytes()
    )
    return None


class VideoWriter(abc.ABC):
    """An abstract base for video writers.

    Define the interface for writing frames to video files; subclasses must
    implement initialization, closing, and frame writing.

    Attributes:
        verbose (bool): Enable verbosity.
        _cur_idx (int): Current written frame index.
        _dst (Path): Destination path for the output video.
        _imgsz (tuple[int, int]): Output video size as (H, W).
        _frame_rate (float): Output video frame rate.
    """
    
    def __init__(
        self,
        dst		  : Path,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float = 30,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        """Initialize the video writer.

        Args:
            dst: Destination path or directory for the video output.
            imgsz: Output video size as (H, W).
            frame_rate: Output video frame rate.
            verbose: Enable verbosity.
        """
        self.verbose     = verbose
        self._cur_idx    = 0
        self._dst        = Path(dst)
        self._imgsz      = I.imgsz(imgsz)
        self._frame_rate = frame_rate
        self._init()
     
     # --- Magic Methods ---
    def __len__(self) -> int:
        """Return the number of written frames."""
        return self._cur_idx
    
    @abc.abstractmethod
    def __del__(self):
        """Close resources held by the writer."""
        pass
    
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
    
    # --- Write  ---
    @abc.abstractmethod
    def write(self, frame: np.ndarray, path: Path = None):
        """Write a frame to the video output.

        Args:
            frame: Frame array as (H, W, C).
            path: Optional path to also save the frame as an image.
        """
        pass


class VideoWriterCV(VideoWriter):
    """A video writer using OpenCV.

    Use cv2.VideoWriter to encode frames and save to disk.
    """
    
    def __init__(
        self,
        dst		  : Path,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float = 30,
        fourcc    : str   = "mp4v",
        verbose   : bool  = False,
        *args, **kwargs
    ):
        """Initialize the OpenCV video writer.

        Args:
            dst: Destination path or directory for the video output.
            imgsz: Output video size as a tuple of (H, W).
            frame_rate: Output video frame rate.
            fourcc: FourCC code for the video codec.
            verbose: Enable verbosity.
        """
        self._fourcc       = fourcc
        self._video_writer = None
        super().__init__(
            dst        = dst,
            imgsz      = imgsz,
            frame_rate = frame_rate,
            verbose    = verbose,
            *args, **kwargs
        )
    
    # ---- Magic Methods ---
    def __del__(self):
        """Close video writer."""
        if self._video_writer:
            self._video_writer.release()
    
    # ---- Initialize ---
    def _init(self):
        """Initialize the OpenCV video writer."""
        if self._dst.is_dir():
            video_file = self._dst / f"result.mp4"
        else:
            video_file = self._dst.parent / f"{self._dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._video_writer = cv2.VideoWriter(
            filename  = str(video_file),
            fourcc    = cv2.VideoWriter_fourcc(*self._fourcc),
            fps       = float(self._frame_rate),
            frameSize =self._imgsz[::-1],  # Must be in [W, H]
            isColor   = True
        )
        
        if self._video_writer is None:
            raise FileNotFoundError(f"``video_file`` cannot be created at {video_file}.")
    
    # --- Write  ---
    def write(self, frame: torch.Tensor | np.ndarray, path: Path = None):
        """Write a frame to the video output.

        Args:
            frame: Video frame as a numpy.ndarray of shape (H, W, C).
            path: Optional path to save frame as image. Defaults to None.
        """
        image = I.to_array(frame)
        # IMPORTANT: Image must be in a BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._video_writer.write(image)
        self._cur_idx += 1


class VideoWriterFFmpeg(VideoWriter):
    """A video writer using FFmpeg.

    Use an ffmpeg subprocess to pipe raw frames to the encoder and save output.
    """
    
    def __init__(
        self,
        dst		  : Path,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float = 10,
        pix_fmt   : str   = "yuv420p",
        verbose   : bool  = False,
        *args, **kwargs
    ):
        """Initialize the FFmpeg video writer.

        Args:
            dst: Destination path or directory for the video output.
            imgsz: Output video size as a tuple of (H, W).
            frame_rate: Output video frame rate.
            pix_fmt: Pixel format for output video.
            verbose: Enable verbosity.
        """
        self._pix_fmt        = pix_fmt
        self._ffmpeg_process = None
        self._ffmpeg_kwargs  = kwargs
        super().__init__(
            dst        = dst,
            imgsz      = imgsz,
            frame_rate = frame_rate,
            verbose    = verbose,
            *args, **kwargs
        )
    
    # ---- Magic Methods ---
    def __del__(self):
        """Close video writer."""
        if self._ffmpeg_process:
            self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.terminate()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
        
    # ---- Initialize ---
    def _init(self):
        """Initialize the FFmpeg video writer."""
        if self._dst.is_dir():
            video_file = self._dst / "result.mp4"
        else:
            video_file = self._dst.parent / f"{self._dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)

        s = f"{self._imgsz[1]}x{self._imgsz[0]}"  # WxH for ffmpeg
        stream = (
            ffmpeg
            .input(
                filename = "pipe:",
                format   = "rawvideo",
                pix_fmt  = "rgb24",
                s        = s
            )
            .output(
                filename = str(video_file),
                pix_fmt  = self._pix_fmt,
                **self._ffmpeg_kwargs
            )
            .overwrite_output()
        )
        if not self.verbose:
            stream = stream.global_args("-loglevel", "quiet")
        self._ffmpeg_process = stream.run_async(pipe_stdin=True)
    
    # --- Write  ---
    def write(self, frame: torch.Tensor | np.ndarray, path: Path = None):
        """Write a frame to the video output.

        Args:
            frame: Video frame as a numpy.ndarray of shape (H, W, C).
            path: Optional path to save frame as image. Defaults to None.
        """
        write_video_ffmpeg(self._ffmpeg_process, frame)
        self._cur_idx += 1
