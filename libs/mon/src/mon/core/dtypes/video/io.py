#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for video input/output operations.

This module provides classes and functions to read and write video files using
different backends such as OpenCV and FFmpeg.
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


# ----- Reading -----
def load_video_ffmpeg(process, height: int, width: int) -> np.ndarray:
    """Reads a frame from video using ``ffmpeg``.
    
    Args:
        process (subprocess.Popen): Subprocess managing ``ffmpeg``.
        height (int): Height of the output frame.
        width (int): Width of the output frame.
        
    Returns:
        numpy.ndarray: Frame/image as a numpy.ndarray of shape (H, W, C) in RGB
            format.
    
    Raises:
        ValueError: If the read bytes length does not match the expected size.
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


# ----- Writing -----
def write_video_ffmpeg(process, frame: Union[torch.Tensor, np.ndarray]):
    """Writes a frame to video using ``ffmpeg``.
    
    Args:
        process (subprocess.Popen): Subprocess managing ``ffmpeg``.
        frame (numpy.ndarray or torch.Tensor): Frame/image as a numpy.ndarray
            of shape (H, W, C) in RGB format.
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
    """A base class for writing images to video.
    
    This class provides an interface for writing video frames to a video file.
    Subclasses must implement the `_init`, `close`, and `write` methods.
    
    Attributes:
        _cur_idx (int): The current index of written frames.
    """
    
    def __init__(
        self,
        dst		  : Path,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float = 30,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        """Initializes the VideoWriter instance.
        
        Args:
            dst (Path): Absolute path to save video. If it is a directory, the
                video will be saved as ``result.mp4``.
            imgsz (tuple[int, int], optional): Output video size as a tuple of
                (H, W). Defaults to (480, 640).
            frame_rate (float, optional): Frame rate of output video. Defaults to 30.
            verbose (bool, optional): Enable verbosity if True. Defaults to False.
        """
        self.verbose     = verbose
        self._cur_idx    = 0
        self._dst        = Path(dst)
        self._imgsz      = I.imgsz(imgsz)
        self._frame_rate = frame_rate
        self._init()
     
     # ----- Magic Methods -----
    def __len__(self) -> int:
        """Returns the number of written frames.
        
        Returns:
            int: Number of written frames.
        """
        return self._cur_idx
    
    @abc.abstractmethod
    def __del__(self):
        """Closes the video writer."""
        pass
    
    # ----- Properties -----
    @property
    def cur_idx(self) -> int:
        """Getter for the current index of written frames.
        
        Returns:
            int: Current index of written frames.
        """
        return self._cur_idx
    
    @property
    def dst(self) -> Path:
        """Getter for the destination path.
        
        Returns:
            Path: Destination path where the video is saved.
        """
        return self._dst
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Getter for the frame size.
        
        Returns:
            tuple[int, int]: Frame size as a tuple of (H, W).
        """
        return self._imgsz
    
    @property
    def frame_rate(self) -> float:
        """Getter for the frame rate.
        
        Returns:
            float: Frame rate of the output video.
        """
        return self._frame_rate
    
    # ----- Initialize -----
    @abc.abstractmethod
    def _init(self):
        """Initializes video writer."""
        pass
    
    # ----- Write  -----
    @abc.abstractmethod
    def write(self, frame: np.ndarray, path: Path = None):
        """Writes a frame to video.

        Args:
            frame (numpy.ndarray): Video frame as a numpy.ndarray of shape
                (H, W, C).
            path (Path, optional): Optional path to save ``frame`` as image.
                Defaults to None.
        """
        pass


class VideoWriterCV(VideoWriter):
    """Writes images to video using ``cv2``.
    
    This class extends VideoWriter and implements video writing using OpenCV's
    VideoWriter.
    
    Attributes:
        _fourcc (str): FourCC code for the video codec.
        _video_writer (cv2.VideoWriter): OpenCV VideoWriter instance.
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
        """Initializes the VideoWriterCV instance.
        
        Args:
            dst (Path): Absolute path to save video. If it is a directory, the
                video will be saved as ``result.mp4``.
            imgsz (tuple[int, int]): Output video size as a tuple of (H, W).
                Defaults to (480, 640).
            frame_rate (float, optional): Frame rate of output video. Defaults to 30.
            fourcc (str, optional): FourCC code for the video codec. Defaults to "mp4v".
            verbose (bool, optional): Enable verbosity if True. Defaults to False.
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
    
    # ---- Magic Methods -----
    def __del__(self):
        """Close video writer."""
        if self._video_writer:
            self._video_writer.release()
    
    # ---- Initialize -----
    def _init(self):
        """Initializes video writer."""
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
    
    # ----- Write  -----
    def write(self, frame: torch.Tensor | np.ndarray, path: Path = None):
        """Writes a frame to video.
        
        Args:
            frame (numpy.ndarray or torch.Tensor): Video frame as a numpy.ndarray
                of shape (H, W, C).
            path (Path, optional): Optional path to save ``frame`` as image.
                Defaults to None.
        """
        image = I.to_array(frame)
        # IMPORTANT: Image must be in a BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._video_writer.write(image)
        self._cur_idx += 1


class VideoWriterFFmpeg(VideoWriter):
    """A class to write images to video using ``ffmpeg``.
    
    This class extends VideoWriter and implements video writing using FFmpeg.
    
    Attributes:
        _pix_fmt (str): Pixel format for output video.
        _ffmpeg_process (subprocess.Popen): Subprocess managing ``ffmpeg``.
        _ffmpeg_kwargs (dict): Additional keyword arguments for FFmpeg output.
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
        """Initializes the VideoWriterFFmpeg instance.
        
        Args:
            dst (Path): Absolute path to save video. If it is a directory, the
                video will be saved as ``result.mp4``.
            imgsz (tuple[int, int]): Output video size as a tuple of (H, W).
                Defaults to (480, 640).
            frame_rate (float): Frame rate of output video. Defaults to 10.
            pix_fmt (str): Pixel format for output video. Defaults to "yuv420p".
            verbose (bool): Enable verbosity if True. Defaults to False.
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
    
    # ---- Magic Methods -----
    def __del__(self):
        """Close video writer."""
        if self._ffmpeg_process:
            self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.terminate()
            self._ffmpeg_process.wait()
            self._ffmpeg_process = None
        
    # ---- Initialize -----
    def _init(self):
        """Initializes video writer."""
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
    
    # ----- Write  -----
    def write(self, frame: torch.Tensor | np.ndarray, path: Path = None):
        """Writes a frame to video.
        
        Args:
            frame (numpy.ndarray or torch.Tensor): Video frame as a numpy.ndarray
                of shape (H, W, C).
            path (Path, optional): Optional path to save ``frame`` as image.
                Defaults to None.
        """
        write_video_ffmpeg(self._ffmpeg_process, frame)
        self._cur_idx += 1
