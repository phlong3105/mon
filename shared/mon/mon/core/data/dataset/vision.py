#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes for vision datasets where image is the primary modality.

We use the ``albumentations`` library for transformations and augmentations.
"""

__all__ = [
    "VisionDataset",
]

import abc

import albumentations as A
import box
import numpy as np
import torch

from mon.core.dtypes.image import Image
from mon.core.enum import Split
from mon.core.pathlib import Path
from .base import BaseDataset, Modalities, Modality

ALBUMENTATIONS_TARGETS = [
    "image",      # The primary input image(s) (e.g., [H, W, C]). Receives geometric, color, and intensity transforms. Uses standard interpolation for geometric transforms.
    "mask",       # Segmentation mask(s) (e.g., [H, W]). Receives geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    "masks",      # Multiple segmentation masks passed together (e.g., [N, H, W]). Processed like mask.
    "bboxes",     # Bounding boxes. Processed according to bbox_params. Requires bbox_params to be set.
    "keypoints",  # Keypoints. Processed according to keypoint_params. Requires keypoint_params to be set.
    "volume",     # A 3D volume (e.g., [D, H, W, C]). Receives 3D geometric transforms, and applicable 2D transforms slice-wise. Color/intensity transforms applied if treated as 'image'.
    "volumes",    # Multiple 3D volumes (e.g., [N, D, H, W, C]). Processed like volume across the first dimension.
    "mask3d",     # A 3D mask (e.g., [D, H, W]). Receives 3D geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    "masks3d"     # Multiple 3D masks (e.g., [N, D, H, W]). Processed like mask3d across the first dimension.
]


# ----- Vision Dataset -----
class VisionDataset(BaseDataset, abc.ABC):
    """Base class for multimodal vision datasets.
    
    Attributes:
        modalities: Dictionary of datapoint modalities.
        
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.TRAIN``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    
    def __init__(
        self,
        root     : Path,
        split    : Split     = Split.TRAIN,
        transform: A.Compose = None,
        verbose  : bool      = True,
        *args, **kwargs
    ):
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # ----- Magic Methods -----
    def __getitem__(self, index: int) -> dict:
        """Retrieves a datapoint and metadata at given ``index`` as a ``dict``."""
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        
        if self.transform:
            pk, _          = self.primary_modality
            args           = {k: v for k, v in datapoint.items() if v is not None}
            args["image"]  = args.pop(pk)
            augmented      = self.transform(**args)
            augmented[pk]  = augmented.pop("image")
            datapoint     |= augmented
            # Convert to float32 if necessary
            for k, v in datapoint.items():
                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                    datapoint[k] = v.to(torch.float32)
                elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                    datapoint[k] = v.astype(np.float32)
                    
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        """Retrieves the total number of datapoints."""
        pk, _ = self.primary_modality
        return len(self.datapoints[pk])
    
    # ----- Initialize -----
    def init_transform(self, transform: A.Compose = None):
        """Initializes transformation operations.

        Args:
            transform: Transformations to apply. Default: ``None``.
        """
        if isinstance(transform, dict | box.Box):
            transform = A.build_compose(**transform)
        if transform is None or isinstance(transform, A.Compose):
            self.transform = transform
        else:
            raise TypeError(f"``transform`` must be None or an instance of "
                            f"albumentations.Compose, got: {type(transform)}.")
        
        # Add additional targets to A.Compose if needed.
        if self.transform:
            additional_targets = {}
            for k, v in self.modalities.items():
                if v.type is None or v.module is None:
                    continue
                if (k not in ALBUMENTATIONS_TARGETS and
                    k not in self.transform.additional_targets):
                    additional_targets[k] = v.type
            if len(additional_targets) > 0:
                self.transform.add_targets(additional_targets=additional_targets)
    
    def reset(self):
        """Resets the dataset to start over."""
        self.index = 0
    
    def close(self):
        """Closes and releases dataset resources."""
        pass
    
    # ----- Data Retrieval -----
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            A ``dict`` containing the datapoint.
        """
        datapoint = {}
        for k, v in self.datapoints.items():
            if hasattr(v[index], "data"):
                datapoint[k] = v[index].data
            else:
                datapoint[k] = None
        return datapoint
    
    def get_meta(self, index: int) -> dict:
        """Gets metadata at the specified ``index``.

        Args:
            index: Index of metadata.

        Returns:
            A ``dict`` containing the metadata.
        """
        pk, _ = self.primary_modality
        return self.datapoints[pk][index].meta
