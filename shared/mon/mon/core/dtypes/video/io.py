#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements I/O operations for videos."""

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
    """Read video frame bytes using ``ffmpeg``.

    Args:
        process: Subprocess managing ``ffmpeg`` instance as ``subprocess.Popen``.
        height: Video frame height.
        width: Video frame width.

    Returns:
        Frame as a ``numpy.ndarray`` of shape :math:`(H, W, C)` in range :math:`[0, 255]`,
        or ``None`` if no data.

    Raises:
        ValueError: If read bytes do not match expected frame size.
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
    """Write frame to video using ``ffmpeg``.

    Args:
        process: Subprocess managing ``ffmpeg`` as ``subprocess.Popen``.
        frame: Frame/image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.

    Raises:
        ValueError: If ``frame`` is not a ``numpy.ndarray``.
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
    """Base class for video writers.

    Args:
        dst: Absolute path to save video. If it is a directory, the video will
            be saved as ``result.mp4``.
        imgsz: Output video size as a ``tuple`` of :math:`(H, W)`. Default: ``(480, 640)``.
        frame_rate: Frame rate of output video. Default: ``30``.
        verbose: Enable verbosity if ``True``. Default: ``False``.
    """
    
    def __init__(
        self,
        dst		  : Path,
        imgsz     : tuple[int, int] = (480, 640),
        frame_rate: float = 30,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        self.dst        = Path(dst)
        self.index      = 0
        self.imgsz      = I.imgsz(imgsz)
        self.frame_rate = frame_rate
        self.verbose    = verbose
        self.init()
        
    def __len__(self) -> int:
        """Returns the number of written frames."""
        return self.index
    
    def __del__(self):
        """Close video writer."""
        self.close()
    
    @abc.abstractmethod
    def init(self):
        """Initialize output handler."""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Close video writer."""
        pass
    
    @abc.abstractmethod
    def write(self, frame: np.ndarray, path: Path = None):
        """Write a frame to video.

        Args:
            frame: Video frame as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.
            path: Optional path to save ``frame`` as image. Default: ``None``.
        """
        pass


class VideoWriterCV(VideoWriter):
    """Write images to video using ``cv2``.

    Args:
        dst: Absolute path to save video. If it is a directory, the video will
            be saved as ``result.mp4``.
        imgsz: Output video size as a ``tuple`` of :math:`(H, W)`. Default: ``(480, 640)``.
        frame_rate: Frame rate of output video. Default: ``30``.
        verbose: Enable verbosity if ``True``. Default: ``False``.
        fourcc: Video codec as ``str``. One of ``"mp4v"``, ``"xvid"``, ``"mjpg"``,
            ``"wmv"``. Default: ``"mp4v"``.
        verbose: Enable verbosity if ``True``. Default: ``False``.
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
        self.fourcc       = fourcc
        self.video_writer = None
        super().__init__(
            dst        = dst,
            imgsz      = imgsz,
            frame_rate = frame_rate,
            verbose    = verbose,
            *args, **kwargs
        )
    
    def init(self):
        """Initialize video writer."""
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_writer = cv2.VideoWriter(
            filename  = str(video_file),
            fourcc    = fourcc,
            fps       = float(self.frame_rate),
            frameSize =self.imgsz[::-1],  # Must be in [W, H]
            isColor   = True
        )
        
        if self.video_writer is None:
            raise FileNotFoundError(f"``video_file`` cannot be created at {video_file}.")
    
    def close(self):
        """Close video writer."""
        if self.video_writer:
            self.video_writer.release()
    
    def write(self, frame: Union[torch.Tensor, np.ndarray], path: Path = None):
        """Write a frame to video.

        Args:
            frame: Video frame as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.
            path: Optional path to save ``frame`` as image. Default: ``None``.
        """
        image = I.to_array(frame)
        # IMPORTANT: Image must be in a BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(image)
        self.index += 1


class VideoWriterFFmpeg(VideoWriter):
    """Write images to video using ``ffmpeg``.

    Args:
        dst: Absolute path to save video. If it is a directory, the video will
            be saved as ``result.mp4``.
        imgsz: Output video size as a ``tuple`` of :math:`(H, W)`. Default: ``(480, 640)``.
        frame_rate: Frame rate of output video. Default: ``30``.
        pix_fmt: Video codec. Default: ``"yuv420p"``.
        verbose: Enable verbosity if ``True``. Default: ``False``.
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
        self.pix_fmt        = pix_fmt
        self.ffmpeg_process = None
        self.ffmpeg_kwargs  = kwargs
        super().__init__(
            dst        = dst,
            imgsz      = imgsz,
            frame_rate = frame_rate,
            verbose    = verbose,
            *args, **kwargs
        )
    
    def init(self):
        """Initialize video writer."""
        if self.dst.is_dir():
            video_file = self.dst / "result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)

        s = f"{self.imgsz[1]}x{self.imgsz[0]}"  # WxH for ffmpeg
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
                pix_fmt  = self.pix_fmt,
                **self.ffmpeg_kwargs
            )
            .overwrite_output()
        )
        if not self.verbose:
            stream = stream.global_args("-loglevel", "quiet")
        self.ffmpeg_process = stream.run_async(pipe_stdin=True)
    
    def close(self):
        """Close video writer."""
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
    
    def write(self, frame: Union[torch.Tensor, np.ndarray], path: Path = None):
        """Write a frame to video.

        Args:
            frame: Video frame as a ``numpy.ndarray`` of shape :math:`(H, W, C)`.
            path: Optional path to save ``frame`` as image. Default: ``None``.
        """
        write_video_ffmpeg(self.ffmpeg_process, frame)
        self.index += 1
