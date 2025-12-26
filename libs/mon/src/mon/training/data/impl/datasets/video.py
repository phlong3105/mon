#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for video-based datasets.

This module implements dataset classes where video/stream data (i.e., frames)
is the primary modality.
"""

__all__ = [
    "VideoLoaderCV",
    "is_video_dataset",
]

from typing import Any

import cv2

from mon.core import log, Path, SAVE_IMAGE_EXT, Split, Task
from mon.core.dtypes import Frame
from mon.training.augment import albumentations as A
from .base import Dataset, Modalities, Modality
from .image import ImageDataset
from ..classes import Classes


# --- Video Loader ---
class VideoLoaderCV(ImageDataset):
    """A concrete class for datasets using OpenCV for a video/stream loading.
    
    This class extends the ``ImageDataset`` class to handle video data as the
    primary modality. It utilizes OpenCV to read video files or streams and
    extract frames on-the-fly during data retrieval.
    
    Attributes:
        _tasks (list[Task]): List of tasks supported by the dataset.
        _modalities (Modalities): A dictionary defining the dataset modalities.
        _num_frames (int): Number of frames in the video.
        _video_capture (cv2.VideoCapture): OpenCV video capture object.
    """
    
    _tasks     : list[Task] = [Task.VIDEO]
    _modalities: Modalities = {
        "frame": Modality(name="image", type="image", module=Frame, train=True, test=True, primary=True),
    }
    
    def __init__(
        self,
        root     : Path,
        split    : Split          = Split.PREDICT,
        transform: A.Compose      = None,
        classes  : Path | Classes = None,
        verbose  : bool           = True,
        *args, **kwargs
    ):
        """Initializes the VideoLoaderCV dataset.
        
        Args:
            root (Path): Path to the video file or stream.
            split (Split): Dataset split type. Defaults to Split.PREDICT.
            transform (A.Compose): Transformations to apply to the data.
                Defaults to None.
            classes (Path or Classes, optional): Either a path to a .yaml file
                containing class label definitions, or a Classes instance.
                If given, this will override any classes defined in the
                subclass. Defaults to None.
            verbose (bool): Whether to print dataset information. Defaults to True.
        """
        self._num_frames    = 0
        self._video_capture = None
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            classes   = classes,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # --- Magic Methods ---
    def __del__(self):
        """Closes the dataset loading mechanism and releases resources."""
        if isinstance(self._video_capture, cv2.VideoCapture):
            self._video_capture.release()
    
    def __iter__(self):
        """Initializes the dataset iterator."""
        self._iter_idx = 0
        if isinstance(self._video_capture, cv2.VideoCapture):
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self
    
    def __len__(self) -> int:
        """Returns the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return self._num_frames
    
    # --- Properties ---
    @property
    def is_stream(self) -> bool:
        """Getter to check if the video source is a stream.
        
        Returns:
            bool: True if the video source is a stream, False otherwise.
        """
        return self._root.is_video_stream() or self._num_frames == -1

    @property
    def shape(self) -> tuple[int, int, int]:
        """Getter for the shape of video frames.
        
        Returns:
            tuple[int, int, int]: A tuple representing the shape of video frames
                as (H, W, C).
        """
        return (
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3
        )
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Getter for the size of video frames.
        
        Returns:
            tuple[int, int]: A tuple representing the size of video frames as (H, W).
        """
        return (
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )
    
    # --- Initialize ---
    def _load_primary_data(self) -> list[Any]:
        """Gets video frames from the ``root`` path.

        Returns:
            list[Any]: An empty list as frames are read on-the-fly.
            
        Raises:
            IOError: If the video source is invalid.
        """
        root = self.root
        if root.is_video_file():
            self._video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif root.is_video_stream():
            self._video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = -1
        else:
            raise IOError(f"Invalid video source: {root}")
        
        if self._num_frames != num_frames:
            self._num_frames = num_frames
        
        return []
    
    def verify(self):
        """Verifies dataset integrity.

        Raises:
            RuntimeError: If no datapoints exist.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset")
        
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    # --- Data Retrieval ---
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Gets a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A dictionary containing the datapoint.

        Raises:
            StopIteration: If the end of the video stream is reached.
            RuntimeError: If the video capture object is not initialized.
        """
        if not self.is_stream and index >= self._num_frames:
            self._video_capture.release()
            raise StopIteration
        
        if isinstance(self._video_capture, cv2.VideoCapture):
            ret_val, frame = self._video_capture.read()
        else:
            raise RuntimeError("``video_capture`` has not been initialized.")
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Frame(data=frame, index=index, path=self._root, root=self._root.parent)
        
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
    
    def _get_meta(self, index: int = 0) -> dict[str, Any]:
        """Gets metadata at the specified ``index``.

        Args:
            index (int): Index of datapoint. Defaults to 0.

        Returns:
            dict[str, Any]: A dictionary containing the metadata.
        """
        path = self.root
        return {
            "index"        : index,
            "path"         : path.parent / path.stem / f"{path.stem}_{index}{SAVE_IMAGE_EXT}",
            "video_path"   : path,
            "orig_shape"   : self.shape,
            "shape"        : self.shape,
            "format"       : self._video_capture.get(cv2.CAP_PROP_FORMAT),
            "fourcc"       : str(self._video_capture.get(cv2.CAP_PROP_FOURCC)),
            "fps"          : int(self._video_capture.get(cv2.CAP_PROP_FPS)),
            "mode"         : self._video_capture.get(cv2.CAP_PROP_MODE),
            "num_frames"   : self._num_frames,
            "pos_avi_ratio": int(self._video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO)),
            "pos_frames"   : int(self._video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
            "pos_msec"     : int(self._video_capture.get(cv2.CAP_PROP_POS_MSEC)),
            "hash"         : path.stat().st_size if isinstance(path, Path) else None,
        }


# --- Validation Check ---
def is_video_dataset(dataset: Dataset) -> bool:
    """Checks if a dataset is a video dataset.

    Args:
        dataset (Dataset): The dataset to check.

    Returns:
        bool: True if the dataset is a video dataset, False otherwise.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "tasks") and isinstance(dataset.tasks, list | tuple):
        return Task.VIDEO in dataset.tasks
    return isinstance(dataset, VideoLoaderCV)
