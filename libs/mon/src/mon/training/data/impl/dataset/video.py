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
        subset: Name of the dataset's subset directory. `Should be defined in
            subclasses`.
        splits: List of supported splits. `Should be defined in subclasses`.
        num_frames: Number of frames in the video.
        shape: Shape of video frames.
        video_capture: OpenCV video capture object.
        video_meta: Video metadata.
        curr_index: Track the last accessed frame to avoid unnecessary seeks.
        transform: Transformations for input and target.
    """

    subset: str | None  = None
    splits: list[Split] = [Split.PREDICT]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root     : Path,
        split    : Split                   = Split.PREDICT,
        transform: A.Compose        | None = None,
        classlist: Path | ClassList | None = None,
        verbose  : bool                    = True,
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
        self.num_frames    = 0
        self.shape         = ()
        self.video_capture = None
        self.video_meta    = {}
        self.curr_index    = -1

        # Continue the initialization chain
        super().__init__(
            root      = root,
            split     = split,
            classlist = classlist,
            verbose   = verbose,
            *args, **kwargs
        )

        # Assign attributes
        self.transform = None
        self.set_transform(value=transform)

    def __del__(self):
        """Close the dataset loading mechanism and release resources."""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.num_frames

    def __iter__(self):
        """Initialize a new iterator."""
        self.curr_index = 0
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the item at the specified ``index``.

        Args:
            index: Index of the item.
        """
        # Fetch datapoint
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.

        transform = self.transform

        if transform is not None:
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
    def set_transform(self, value: Any):
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

        self.transform = value

    @property
    def is_stream(self) -> bool:
        """Check if the video source is a stream."""
        return self.root.is_video_stream() or self.num_frames == -1

    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the size of video frames."""
        return (
            int(self.shape[0]),
            int(self.shape[1])
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
        root = self.root

        self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)

        if not self.video_capture.isOpened():
            raise RuntimeError(f"Failed to open video source at: {root}")

        # Cache values to avoid repeated C-calls
        self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.shape = (h, w, 3)

        # Retrieve video metadata
        self.video_meta = {
            "video_path"   : root,
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
        if not self.video_capture or not self.video_capture.isOpened():
            raise RuntimeError(f"VideoCapture is not initialized.")

        # Smart Seeking
        # Only seek if the requested index is NOT the next sequential frame
        if index != self.curr_index + 1:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = self.video_capture.read()

        if not success:
            if self.is_stream:
                raise StopIteration
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        self.curr_index = index

        # Format Conversion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in your Frame/Image object (assuming it handles metadata)
        frame_obj = Frame(data=frame, index=index, path=self.root, root=self.root.parent)

        # Build datapoint dictionary
        path = self.root
        meta = {
            "index": index,
            "path" : path.parent / path.stem / f"{path.stem}_{index}{EXT.IMAGE}",
        } | self.video_meta
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
