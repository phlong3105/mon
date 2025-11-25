#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements dataset classes where video/stream data (i.e., frames)
is the primary modality.
"""

__all__ = [
    "VideoLoader",
    "VideoLoaderCV",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "is_video_dataset",
]

import abc
from typing import Union

import cv2
import ffmpeg
import numpy as np
import torch

from mon.core import (
    image as I,
    log,
    Path,
    SAVE_IMAGE_EXT,
    Split,
    Task,
    video as V,
)
from mon.core.dtypes import Frame
from mon.training.augment import albumentations as A
from .base import BaseDataset, Modalities, Modality
from .vision import VisionDataset


# ----- Video Loader -----
class VideoLoader(VisionDataset, abc.ABC):
    """Base class for video loaders.

    Attributes:
        tasks: List of supported tasks.
        modalities: Dictionary of datapoint modalities.

    Args:
        root: Absolute path to the video file or stream.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.PREDICT``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    tasks     : list[Task]  = [Task.VIDEO]
    modalities: Modalities  = {
        "frame": Modality(name="image", type="image", module=Frame, train=True, test=True, primary=True),
    }
    
    def __init__(
        self,
        root      : Path,
        split     : Split     = Split.PREDICT,
        transform : A.Compose = None,
        verbose   : bool      = True,
        *args, **kwargs
    ):
        self.num_frames = 0
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # ----- Magic Methods -----
    def __len__(self) -> int:
        """Retrieves the number of frames in the video."""
        return self.num_frames
    
    # ----- Initialize -----
    def verify_data(self):
        """Verifies dataset integrity.

        Raises:
            RuntimeError: If no datapoints exist.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset")
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")


class VideoLoaderCV(VideoLoader):
    """Loads video frames from a file or stream using ``cv2``.

    Args:
        root: Absolute path to the video file or stream.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.PREDICT``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    def __init__(
        self,
        root     : Path,
        split    : Split     = Split.PREDICT,
        transform: A.Compose = None,
        verbose  : bool      = True,
        *args, **kwargs
    ):
        self.video_capture = None
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # ----- Properties -----
    @property
    def is_stream(self) -> bool:
        """Returns ``True`` if the input is a stream, ``False`` otherwise."""
        return self.root.is_video_stream() or self.num_frames == -1

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the shape of video frames as a tuple of
        :math:`(height, width, channels)`.
        """
        return (
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3
        )
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Returns the resolution of video frames as a tuple of
        :math:`(height, width)`.
        """
        return (
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )
    
    # ----- Initialize -----
    def list_primary_data(self) -> list:
        """Gets video frames from the ``root`` path.

        Raises:
            IOError: If ``root`` is not a valid video file or stream.
        """
        root = Path(self.root)
        if root.is_video_file():
            self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif root.is_video_stream():
            self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = -1
        else:
            raise IOError(f"Invalid video source: {self.root}")
        
        if self.num_frames != num_frames:
            self.num_frames = num_frames
        
        return []
        
    def reset(self):
        """Resets the video loader."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """Closes and releases video capture."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.release()
    
    # ----- Data Retrieval -----
    def get_datapoint(self, index: int) -> dict:
        """ets a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            A ``dict`` containing the datapoint.

        Raises:
            StopIteration: If index exceeds frame count for non-streams.
            RuntimeError: If ``video_capture`` not initialized.
        """
        if not self.is_stream and index >= self.num_frames:
            self.close()
            raise StopIteration
        
        if isinstance(self.video_capture, cv2.VideoCapture):
            ret_val, frame = self.video_capture.read()
        else:
            raise RuntimeError("[video_capture] has not been initialized.")
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Frame(data=frame, index=index, orig_shape=self.imgsz, path=self.root)
        
        pk, _     = self.primary_modality
        datapoint = {}
        for k, v in self.datapoints.items():
            if k == pk:
                datapoint[k] = frame.data
            elif v is not None and v[index] and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
            else:
                datapoint[k] = None

        return datapoint
    
    def get_meta(self, index: int = 0) -> dict:
        """Gets metadata at the specified ``index``.

        Args:
            index: Index of metadata. Default: ``0``.

        Returns:
            A ``dict`` containing the metadata.
        """
        path = self.root
        return {
            "index"        : index,
            "path"         : path.parent / path.stem / f"{path.stem}_{index}{SAVE_IMAGE_EXT}",
            "video_path"   : path,
            "orig_shape"   : self.shape,
            "shape"        : self.shape,
            "format"       : self.video_capture.get(cv2.CAP_PROP_FORMAT),
            "fourcc"       : str(self.video_capture.get(cv2.CAP_PROP_FOURCC)),
            "fps"          : int(self.video_capture.get(cv2.CAP_PROP_FPS)),
            "mode"         : self.video_capture.get(cv2.CAP_PROP_MODE),
            "num_frames"   : self.num_frames,
            "pos_avi_ratio": int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO)),
            "pos_frames"   : int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
            "pos_msec"     : int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC)),
            "hash"         : self.root.stat().st_size if isinstance(self.root, Path) else None,
        }


# ----- Video Writer -----
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
        V.write_video_ffmpeg(self.ffmpeg_process, frame)
        self.index += 1


# ----- Validation Check -----
def is_video_dataset(dataset: BaseDataset) -> bool:
    """Checks if a dataset is a video dataset.

    Args:
        dataset: Dataset to check.

    Returns:
        ``True`` if dataset is a video dataset, ``False`` otherwise.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "tasks") and isinstance(dataset.tasks, list | tuple):
        return Task.VIDEO in dataset.tasks
    return isinstance(dataset, VideoLoader | VideoLoaderCV)
