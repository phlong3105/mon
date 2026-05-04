#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video Datasets.

This module provides video-based datasets.
"""

from __future__ import annotations

__all__ = [
    "VideoOnlyDataset",
    "is_video_dataset",
]

from typing import Any, override

import cv2

from mon.core import (
    build_classlist,
    ClassList,
    Frame,
    MetadataDictList,
    Path,
    Size,
    Split,
)
from mon.dataset.transform import build_compose, Compose
from .dataset import Dataset, StandardDataset
from .image import AlbumentationsDataset
from .modality import build_modalities, FrameModality, ModalityList


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class VideoOnlyDataset(StandardDataset, AlbumentationsDataset):
    """Standard video-only dataset.

    Extend the base ``StandardDataset`` class with ``AlbumentationsDataset`` to
    support single-video datasets with multiple splits and albumentations
    transformations.
    """

    dirname: str = ""
    subdir: str = ""
    splits: list[Split] = [Split.PREDICT]
    modalities: ModalityList = ModalityList([
        FrameModality(name="image", dirname=""),
    ])
    classes: ClassList = ClassList()

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: Path,
        split: Split = Split.PREDICT,
        dirname: str = "",
        subdir: str = "",
        transforms: Compose | None = None,
        keep_original: bool = False,
        modalities: ModalityList | None = None,
        classes: ClassList | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (Path): Path to the root directory of the dataset.
            split (Split, optional): Data split subset to use. Must be one of
                the options defined in ``splits``.
            dirname (str, optional): Name of the dataset directory within the
                root path. Use this if the given ``root`` path does not contain
                the dataset directory itself. Defaults to "".
            subdir (str, optional): Name of the subdirectory within the dataset's
                ``root`` (i.e., ``root/subdir``). Use this if the current
                dataset is a subset of another dataset. If provided, it
                overrides the class-level default. Defaults to "".
            transforms (Compose | None, optional): Transformations to apply.
                Defaults to None.
            keep_original (bool, optional): Whether to keep the original data
                in the datapoint dictionary. If True, the original data will be
            modalities (ModalityList | None, optional): A list of ``Modality``
                definitions. By default, the first modality is considered the
                primary one. If provided, it overrides the class-level default.
                Defaults to None.
            classes (ClassList | None, optional): Class definitions associated
                with the dataset. If provided, it overrides the class-level
                default. Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        # Allocate resources
        self.video_capture = None
        self.video_meta = {}
        self.curr_index = -1

        # Continue the initialization chain
        super().__init__(
            root=root,
            split=split,
            dirname=dirname,
            subdir=subdir,
            transforms=transforms,
            keep_original=keep_original,
            modalities=modalities,
            classlist=classes,
            verbose=verbose,
            *args, **kwargs
        )

    @override
    def __del__(self):
        """Finalizer called when the object is about to be destroyed."""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()

    # --- Container / Sequence Methods ---
    @override
    def __len__(self) -> int:
        """Return the length of the container."""
        return self.video_meta["num_frames"]

    @override
    def __iter__(self):
        """Return an iterator for the container."""
        self.curr_index = 0
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    # --- Properties ---
    @property
    def is_stream(self) -> bool:
        """Check if the video source is a stream."""
        return self.root.is_url() or self.video_meta["num_frames"] == -1

    @property
    def imgsz(self) -> Size:
        """Return the size of video frames."""
        return self.video_meta["imgsz"]

    # --- Discovery ---
    @override
    def list_metapoints(self):
        """Setup ``self.video_capture`` and ``self.video_meta``.

        Set ``self.metapoints`` empty since the datapoints (i.e., frames) are
        loaded on-the-fly via ``get_datapoint()``.

        Raises:
            RuntimeError: If the video source cannot be opened.
        """
        # Initialize empty metapoints dictionary with modalities
        self.metapoints = MetadataDictList.from_keys(self.modalities.keys)

        # Setup video capture
        src = self.base_dir
        self.video_capture = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)

        if not self.video_capture.isOpened():
            raise RuntimeError(f"Failed to open video source at: {src}")

        # Retrieve video metadata
        h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_meta = {
            "video_path": src,
            "shape": (h, w, 3),
            "imgsz": Size(height=h, width=w),
            "format": self.video_capture.get(cv2.CAP_PROP_FORMAT),
            "fourcc": str(self.video_capture.get(cv2.CAP_PROP_FOURCC)),
            "fps": int(self.video_capture.get(cv2.CAP_PROP_FPS)),
            "mode": self.video_capture.get(cv2.CAP_PROP_MODE),
            "num_frames": int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "pos_avi_ratio": int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO)),
            "pos_frames": int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
            "pos_msec": int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC)),
            "hash": src.stat().st_size if isinstance(src, Path) else None,
        }

    # --- Creation ---
    @override
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "VideoOnlyDataset":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        transforms = config.pop("transforms", None)
        modalities = config.pop("modalities", None)
        classes = config.pop("classes", None)

        # Build the objects
        transforms = build_compose(transforms)
        modalities = build_modalities(modalities)
        classes = build_classlist(classes)

        # Return the new instance
        config |= kwargs
        return cls(
            transforms=transforms,
            modalities=modalities,
            classes=classes,
            **config
        )

    # --- Retrieval ---
    @override
    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        # 1. Validate video capture
        if not self.video_capture or not self.video_capture.isOpened():
            raise RuntimeError(f"VideoCapture is not initialized.")

        # 2. Retrieve frame
        # Only seek if the requested index is NOT the next sequential frame
        if index != self.curr_index + 1:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = self.video_capture.read()

        if not success:
            if self.is_stream:
                raise StopIteration
            raise IndexError(
                f"Index {index} out of range for video of length {len(self)}."
            )

        self.curr_index = index

        # 3. Wrap the frame into the corresponding structure
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Frame(
            image=frame,
            index=index,
            path=self.base_dir,
            base_dir=self.base_dir.parent
        )

        # 4. Build datapoint dictionary
        pk, pm = self.primary
        ext = pm.ext
        src = self.base_dir
        meta = {
            "index": index,
            "path": src.parent / src.stem / f"{src.stem}_{index:06d}{ext}",
        } | self.video_meta

        datapoint = {pk: frame, "meta":meta}
        return datapoint

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def is_video_dataset(dataset: Dataset | None) -> bool:
    """Check if a dataset is a video dataset.

    Args:
        dataset (Dataset | None): Dataset to check.

    Returns:
        bool: True if the dataset is a video dataset, False otherwise.
    """
    if dataset is None:
        return False
    return isinstance(dataset, VideoOnlyDataset)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
