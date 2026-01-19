#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video-based datasets.

This module provides base classes for video datasets and data loaders.
"""

from __future__ import annotations

__all__ = [
    "VideoLoader",
    "is_video_dataset",
]

from typing import Any

import box
import cv2
import numpy as np
import torch

from mon.core import log, Path, EXT, Split, Task
from mon.core.dtypes import ClassList, Frame
from mon.training.augment import albumentations as A
from ...base import Dataset
from ...comp import BatchCollateMixin, RootLoadMixin


# ==============================================================================
# region VIDEO DATASETS
# ==============================================================================

class VideoLoader(Dataset, RootLoadMixin, BatchCollateMixin):
    """Video or stream loader using OpenCV.

    Extend ``Dataset`` to handle video data as the primary modality. Use OpenCV
    to read video files or streams and extract frames on-the-fly during data
    retrieval.

    Attributes:
        _num_frames (int): Number of frames in the video.
        _shape (tuple): Shape of video frames.
        _video_capture (cv2.VideoCapture): OpenCV video capture object.
        _video_meta (dict): Video metadata.
        _transform (albumentations.Compose | None): Transformations for input and target.
        _curr_index (int): Track last accessed frame to avoid unnecessary seeks.

    Attributes:
        _subset (str | None): Name of the dataset's subset directory.
            Defaults to None.
        _splits (list[Split]): List of supported splits. Defaults to [].
    """

    _subset: str | None  = None
    _splits: list[Split] = [Split.PREDICT]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root     : Path,
        split    : Split                    = Split.PREDICT,
        transform: A.Compose | None         = None,
        classlist: Path | ClassList | None  = None,
        verbose  : bool                     = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use. Defaults to Split.PREDICT.
            transform: Transformations to apply. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ClassList instance. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Define default values to avoid potential attribute errors during
        # the initialization chain
        self._num_frames    = 0
        self._shape         = ()
        self._video_capture = None
        self._video_meta    = {}
        self._curr_index    = -1

        # Continue the initialization chain
        super().__init__(
            root      = root,
            split     = split,
            classlist = classlist,
            verbose   = verbose,
            *args, **kwargs
        )
        self.transform = transform

    def __del__(self):
        """Close the dataset loading mechanism and release resources."""
        if self._video_capture and self._video_capture.isOpened():
            self._video_capture.release()

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._num_frames

    def __iter__(self):
        """Initialize a new iterator."""
        self._curr_index = 0
        if isinstance(self._video_capture, cv2.VideoCapture):
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the item at the specified ``index``.

        Args:
            index: Index of the item.
        """
        # Fetch datapoint
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.

        transform = self._transform

        if transform:
            # Albumentations usually uses the key 'image'
            augmented     = transform(image=data["frame"])
            data["frame"] = augmented["image"]

            # Vectorized-style type casting
            for k, v in data.items():
                if v is not None:
                    # Converts non‑float tensors/arrays to float32
                    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                        data[k] = v.to(torch.float32)
                    elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                        data[k] = v.astype(np.float32)

        return {**data, "meta": meta}

    # --- Properties ---
    @property
    def transform(self) -> A.Compose | None:
        """Return the transformation operations."""
        return self._transform

    @transform.setter
    def transform(self, value: Any):
        """Set the transformation operations.

        Args:
            value: Transformations for input and target.

        Raises:
            TypeError: If ``value`` is not an instance of albumentations.Compose.
        """
        if isinstance(value, (dict, box.Box)):
            value = A.Compose(**value)
        if value is not None and not isinstance(value, A.Compose):
            raise TypeError(
                f"Expected 'transform' to be an instance of albumentations.Compose, "
                f"but got {type(value).__name__}."
            )

        self._transform = value

    @property
    def is_stream(self) -> bool:
        """Check if the video source is a stream."""
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
        """Load core data for the dataset.

        Returns:
            Empty dictionary as frames are loaded on-the-fly.

        Raises:
            RuntimeError: If the video source cannot be opened.
        """
        # Validate video source
        root = self._root

        self._video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)

        if not self._video_capture.isOpened():
            raise RuntimeError(f"Failed to open video source at: {root}")

        # Cache values to avoid repeated C-calls
        self._num_frames = int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        h = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._shape = (h, w, 3)

        # Retrieve video metadata
        self._video_meta = {
            "video_path"   : root,
            "orig_shape"   : self._shape,
            "shape"        : self._shape,
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
            RuntimeError: If no datapoints are found.
        """
        if len(self) == 0:
            raise RuntimeError(f"No datapoints in the dataset: {self.__class__.__name__}.")

        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {len(self)}.")

    # --- Data Retrieval ---
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Raises:
            RuntimeError: If ``VideoCapture`` is not initialized.
            IndexError: If frame at ``index`` could not be read.
        """
        if not self._video_capture or not self._video_capture.isOpened():
            raise RuntimeError(f"VideoCapture is not initialized.")

        # Smart Seeking
        # Only seek if the requested index is NOT the next sequential frame
        if index != self._curr_index + 1:
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = self._video_capture.read()

        if not success:
            if self.is_stream:
                raise StopIteration
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        self._curr_index = index

        # Format Conversion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in your Frame/Image object (assuming it handles metadata)
        frame_obj = Frame(data=frame, index=index, path=self.root, root=self._root.parent)

        # Build datapoint dictionary
        path = self._root
        meta = {
            "index": index,
            "path" : path.parent / path.stem / f"{path.stem}_{index}{EXT.IMAGE}",
        } | self._video_meta
        return {"frame": frame_obj, "meta": meta}

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def is_video_dataset(dataset: Dataset) -> bool:
    """Check if a dataset is a video dataset.

    Args:
        dataset: Dataset to check.

    Returns:
        True if the dataset is a video dataset, False otherwise.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "tasks") and isinstance(dataset.tasks, list | tuple):
        return Task.VIDEO in dataset.tasks
    return isinstance(dataset, VideoLoader)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
