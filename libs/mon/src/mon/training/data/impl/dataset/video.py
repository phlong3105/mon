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

from typing import Any, override

import box
import cv2
import numpy as np
import torch

from mon.core import EXT, log, Path, Split, Task
from mon.core.types import ClassList, Frame
from mon.training.augment import albumentations as A
from ...base import Dataset
from ...comp import BatchCollateMixin, RootLoadMixin


# ==============================================================================
# region VIDEO DATASETS
# ==============================================================================

class VideoLoader(Dataset, RootLoadMixin, BatchCollateMixin):
    """Video or stream loader using OpenCV.

    Extend ``Dataset`` with ``RootLoadMixin`` and ``BatchCollateMixin`` to
    load video data from a specified ``root``. Use OpenCV to read video frames
    on-the-fly.

    Attributes:
        subroot (str, optional): Name of the subdirectory within the dataset's
            ``root`` (i.e., ``root/subroot``). Use this if the current dataset
            is a subset of another dataset. `Should be defined in subclasses.`
        splits (list[Split]): List of supported splits. `Must be defined in
            subclasses.`
    """

    subroot: str = ""
    splits: list[Split] = [Split.PREDICT]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: Path | str,
        split: Split | str = Split.TRAIN,
        transform: A.Compose | dict | None = None,
        classlist: ClassList | Path | str | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (Path | str): Absolute path to the dataset root directory.
            split (Split | str): Data split subset to use. Must be one of the
                supported ``splits``. Defaults to Split.PREDICT.
            transform (A.Compose | dict, optional): Transformations to apply.
                Defaults to None.
            classlist (ClassList | Path | str, optional): Class definitions for
                the dataset. Can be a ``ClassList`` instance or a path to a
                .yaml file. Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.
        """
        # Define default values to avoid potential attribute errors during
        # the initialization chain
        self.video_capture = None
        self.meta = box.Box()
        self.curr_index = -1

        # Continue the initialization chain
        super().__init__(
            root=root,
            split=split,
            classlist=classlist,
            verbose=verbose,
            *args, **kwargs,
        )

        # Assign attributes
        self.transform = transform

    @override
    def __del__(self):
        """Finalize the object.

        Close the dataset loading mechanism and release resources.
        """
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()

    # --- Container / Sequence Methods ---
    @override
    def __len__(self) -> int:
        """Return the length of the container."""
        return self.meta.num_frames

    @override
    def __iter__(self):
        """Return an iterator for the container."""
        self.curr_index = 0
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        # Fetch datapoint
        data = self.get_underlying_data(index=index)
        meta = data.pop("meta")

        transform = self.transform

        if transform:
            augmented = transform(image=data["frame"])
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
        """Return the transformation pipeline."""
        return self._transform

    @transform.setter
    def transform(self, transform: A.Compose | dict | None):
        """Set the transformation operations.

        Args:
            transform (A.Compose | dict, optional): Transformations to apply.
                Defaults to None.

        Raises:
            TypeError: If ``transform`` is not an instance of albumentations.Compose.
        """
        if transform is None:
            self._transform = None
            return

        if isinstance(transform, dict):
            transform = A.Compose(**transform)
        if not isinstance(transform, A.Compose):
            raise TypeError(
                f"Expected 'transform' to be an instance of "
                f"albumentations.Compose, but got {type(transform).__name__}."
            )

        self._transform = transform

    @property
    def is_stream(self) -> bool:
        """Check if the video source is a stream."""
        return self.root.is_video_stream() or self.meta.num_frames == -1

    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the size of video frames."""
        return self.meta.imgsz

    # --- Initialize ---
    @override
    def _load_data(self) -> dict[str, list[Any]]:
        """Load the core data of the dataset.

        Returns:
            dict[str, list[Any]]: Dictionary containing lists of datapoints for
                each modality.

        Raises:
            RuntimeError: If ``video_capture`` cannot be opened.
        """
        # Validate video source
        root = self.root

        self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)

        if not self.video_capture.isOpened():
            raise RuntimeError(f"Failed to open video source at: {root}")

        h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Retrieve video metadata
        self.meta = box.Box({
            "video_path": root,
            "shape": (h, w, 3),
            "imgsz": (h, w),
            "format": self.video_capture.get(cv2.CAP_PROP_FORMAT),
            "fourcc": str(self.video_capture.get(cv2.CAP_PROP_FOURCC)),
            "fps": int(self.video_capture.get(cv2.CAP_PROP_FPS)),
            "mode": self.video_capture.get(cv2.CAP_PROP_MODE),
            "num_frames": int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "pos_avi_ratio": int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO)),
            "pos_frames": int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
            "pos_msec": int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC)),
            "hash": root.stat().st_size if isinstance(root, Path) else None,
        }, frozen_box=True)

        # Return empty dict since frames are loaded on-the-fly via the
        # ``_get_datapoint()`` method.
        return {}

    @override
    def verify(self):
        """Verify dataset integrity.

        Raises:
            RuntimeError: If no datapoints are found.
        """
        if len(self) == 0:
            raise RuntimeError(
                f"No datapoints in the dataset: {self.__class__.__name__}."
            )

        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {len(self)}.")

    # --- Data Retrieval ---
    @override
    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.

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
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}."
            )

        self.curr_index = index

        # Format Conversion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in your Frame/Image object (assuming it handles metadata)
        frame_obj = Frame(
            data=frame,
            index=index,
            path=self.root,
            root=self.root.parent
        )

        # Build datapoint dictionary
        path = self.root
        meta = {
            "index": index,
            "path": path.parent / path.stem / f"{path.stem}_{index}{EXT.IMAGE}",
        } | self.meta
        return {
            "frame": frame_obj,
            "meta": meta,
        }

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def is_video_dataset(dataset: Dataset) -> bool:
    """Check if a dataset is a video dataset.

    Args:
        dataset (Dataset): Dataset to check.

    Returns:
        bool: True if the dataset is a video dataset, False otherwise.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "tasks") and isinstance(dataset.tasks, (list, tuple)):
        return Task.VIDEO in dataset.tasks
    return isinstance(dataset, VideoLoader)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
