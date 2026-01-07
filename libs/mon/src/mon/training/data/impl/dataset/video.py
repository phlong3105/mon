#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video-based datasets.

This module provides base classes for video datasets and data loaders, including
functionality for loading videos and applying transformations.
"""

__all__ = [
    "VideoLoader",
    "is_video_dataset",
]

from typing import Any

import box
import cv2
import numpy as np
import torch

from mon.core import log, Path, SAVE_IMAGE_EXT, Split, Task
from mon.core.dtypes import ClassList, Frame
from mon.training.augment import albumentations as A
from ...base import Dataset
from ...comp import BatchCollateMixin, RootLoadMixin


# ==============================================================================
# LOADERS
# ==============================================================================

class VideoLoader(Dataset, RootLoadMixin, BatchCollateMixin):
    """A concrete class for loading a single video or stream using OpenCV.
    
    Extend the ``ImageDataset`` to handle video data as the primary modality.
    Use OpenCV to read video files or streams and extract frames on-the-fly
    during data retrieval.
    
    Attributes:
        _num_frames (int): Number of frames in the video.
        _video_capture (cv2.VideoCapture): OpenCV video capture object.
        _video_meta (dict): Video metadata.
        _transform (albumentations.Compose): Transformations for input/target.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root     : Path,
        split    : Split            = Split.PREDICT,
        transform: A.Compose        = None,
        classlist: Path | ClassList = None,
        verbose  : bool             = True,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use. One of: Split.TRAIN, Split.VAL,
                Split.TEST, or Split.PREDICT. Defaults to Split.TRAIN.
            transform: Transformations to apply. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ``ClassList`` instance. If given, this will override any
                ``classes`` defined in the subclass. Defaults to None.
            verbose: If True, enables verbose output. Defaults to True.
        """
        self._num_frames    = 0
        self._video_capture = None
        self._video_meta    = {}
        super().__init__(
            root      = root,
            split     = split,
            classlist = classlist,
            verbose   = verbose,
            *args, **kwargs
        )
        self.transform = transform
    
    def __del__(self):
        """Close the dataset loading mechanism and releases resources."""
        if isinstance(self._video_capture, cv2.VideoCapture):
            self._video_capture.release()
    
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the dataset (i.e., number of frames)."""
        return self._num_frames
    
    def __iter__(self):
        """Initialize a new iterator."""
        self._iter_idx = 0
        if isinstance(self._video_capture, cv2.VideoCapture):
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the datapoint at the specified ``index`` in ``_datapoints``.
        
        Args:
            index: Index of datapoint.
            
        Returns:
            A dictionary containing the datapoint and its metadata.
        """
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.
        
        if self.transform:
            augmented     = self.transform(image=data["frame"])
            data["frame"] = augmented["frame"]
            # Convert to float32 if necessary
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                    data[k] = v.to(torch.float32)
                elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                    data[k] = v.astype(np.float32)
                    
        return data | {"meta": meta}
    
    # --- Properties ---
    @property
    def transform(self) -> A.Compose:
        """Return the transformation operations."""
        return self._transform
    
    @transform.setter
    def transform(self, transform: Any):
        """Setter for transformation operations.
        
        Args:
            transform: Transformations for input/target.
            
        Raises:
            TypeError: If ``transform`` is not None or an instance of
                albumentations.Compose.
        """
        if isinstance(transform, dict | box.Box):
            transform = A.Compose(**transform)
        if transform is not None and not isinstance(transform, A.Compose):
            raise TypeError(f"``transform`` must be None or an instance of "
                            f"albumentations.Compose, got: {type(transform)}.")

        self._transform = transform
    
    @property
    def is_stream(self) -> bool:
        """Return True if the video source is a stream, False otherwise."""
        return self._root.is_video_stream() or self._num_frames == -1

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of video frames."""
        return (
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3
        )
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the size of video frames."""
        return (
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )
    
    # --- Initialize ---
    def _load_data(self) -> dict[str, Any]:
        """Core data loading mechanism for the dataset."""
        # Validate video source
        root = self.root
        if root.is_video_file():
            self._video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif root.is_video_stream():
            self._video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
            num_frames = -1
        else:
            raise IOError(f"Invalid video source: {root}")
        
        # Set number of frames
        if self._num_frames != num_frames:
            self._num_frames = num_frames
        
        # Retrieve video metadata
        self._video_meta = {
            "video_path"   : root,
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
            "hash"         : root.stat().st_size if isinstance(root, Path) else None,
        }
        
        # Return empty dict since frames are loaded on-the-fly via the
        # ``_get_datapoint()`` method.
        return {}
    
    def verify(self):
        """Verify dataset integrity.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() == 0:
            raise RuntimeError("No datapoints in the dataset")
        
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    # --- Data Retrieval ---
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            A dictionary containing all modalities for the specified datapoint.

        Raises:
            StopIteration: If the end of the video stream is reached.
            RuntimeError: If the video capture object is not initialized.
        """
        # Validate video reader
        if not self.is_stream and index >= self._num_frames:
            self._video_capture.release()
            raise StopIteration
        
        # Read raw data from video capture
        if isinstance(self._video_capture, cv2.VideoCapture):
            ret_val, frame = self._video_capture.read()
        else:
            raise RuntimeError("``video_capture`` has not been initialized.")
        
        # Wrap raw data into a Frame object
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Frame(data=frame, index=index, path=self._root, root=self._root.parent)
        
        # Build datapoint dictionary
        path      = self.root
        meta      = {
            "index": index,
            "path" : path.parent / path.stem / f"{path.stem}_{index}{SAVE_IMAGE_EXT}",
        } | self._video_meta
        datapoint = {"frame": frame, "meta": meta}
        return datapoint


# ==============================================================================
# UTILITIES
# ==============================================================================

# --- Validation & Sanitization ---
def is_video_dataset(dataset: Dataset) -> bool:
    """Check if a dataset is a video dataset.

    Args:
        dataset: The dataset to check.

    Returns:
        True if the dataset is a video dataset, False otherwise.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "tasks") and isinstance(dataset.tasks, list | tuple):
        return Task.VIDEO in dataset.tasks
    return isinstance(dataset, VideoLoader)
