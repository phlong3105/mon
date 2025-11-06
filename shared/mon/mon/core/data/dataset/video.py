#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video-based Datasets.

This module implements dataset classes where video data (i.e., frames) is the
primary modality.
"""

__all__ = [
    "VideoLoader",
    "VideoLoaderCV",
    "is_video_dataset",
]

import abc

import albumentations as A
import cv2

from mon.constants import SAVE_IMAGE_EXT
from mon.core.console import log
from mon.core.dtypes.video import Frame
from mon.core.enum import Split, Task
from mon.core.pathlib import Path
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
