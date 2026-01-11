#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image-based datasets.

This module implements base classes for image datasets and data loaders,
including functionality for loading images and applying transformations.
"""

__all__ = [
    "ImageDataset",
    "ImageLoader",
]

import glob
from typing import Any

import box
import numpy as np
import torch

from mon.core import create_progress_bar, log, Path, Split
from mon.core.dtypes import ClassList, Image
from mon.training.augment import albumentations as A
from ...base import Dataset, Modalities, Modality
from ...comp import BatchCollateMixin, MultimodalDataLoadMixin


# ==============================================================================
# DATASETS
# ==============================================================================

class ImageDataset(
    Dataset,
    MultimodalDataLoadMixin,
    BatchCollateMixin
):
    """A base class for datasets where images are the primary modality.
    
    This class defines a concrete implementation for image-based datasets. It
    extends ``Dataset`` with mixins for metadata handling, multimodal data
    loading, and DataLoader's functionality. It also defines transformation
    operations using the albumentations library. For vision tasks, this is the
    primary dataset class to extend from.
    
    Attributes:
        _subset (str): The name of the dataset's subset directory. Since the
            given attribute ``root`` may only set the dataset root directory,
            this attribute defines the actual folder name of the sub-dataset
            within the root directory (e.g., dataset with multiple versions).
            Defaults to None and should be overridden in subclasses.
        _splits (list[Split]): A list of supported splits. This is used to
            validate the given attribute ``split``. Defaults to all four splits:
            Split.TRAIN, Split.VAL, Split.TEST, and Split.PREDICT. Should be
            overridden in subclasses if needed.
        _modalities (Modalities): A dictionary defining the dataset modalities.
            Should be overridden in subclasses to accommodate additional
            modalities (e.g., depth maps, segmentation masks, bounding boxes,
            captions, or other sensor data).
        _classlist (ClassList): The dataset object classes. Defaults to None and
            should be overridden in subclasses.
        _transform (albumentations.Compose): Transformations to apply to the data.
        verbose (bool): If True, enables verbose output.
    """
    
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classlist : ClassList   = None
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root     : Path,
        split    : Split            = Split.TRAIN,
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
        
        Raises:
            ValueError: If ``modalities`` has no defined attributes.
        """
        # Validate modalities
        if not self._modalities:
            raise ValueError("Expected 'modalities' to have at least one attribute,"
                             " but got empty.")
        
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
        """Close the dataset loading mechanism and releases resources."""
        pass
    
    # --- Representation ---
    def __repr__(self) -> str:
        """Official string representation for developers (eval-able)."""
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        if self._root:
            lines += [f"Root location: {self._root}"]
        if self._transform:
            lines += [repr(self._transform)]
        return "\n".join(lines)
    
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container (i.e., number of datapoints)."""
        # Optimization: Use the internal dict directly to avoid property overhead
        # and search only for the primary modality key once.
        pk = next(k for k, v in self._modalities.items() if v.primary)
        return len(self._datapoints[pk])
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Define behavior for when an item is accessed via the notation self[index]."""
        # Fetch datapoint
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.
        
        transform = self._transform
        
        if transform:
            pk, _ = self.primary_modality
            
            # Albumentations expects 'image'. We map pk -> 'image' without pop/merge overhead.
            if pk != "image":
                data["image"] = data.pop(pk)
            
            # Filter None values efficiently
            augmented = transform(**{k: v for k, v in data.items() if v is not None})
            
            # Revert 'image' back to the primary modality key if necessary
            if pk != "image":
                augmented[pk] = augmented.pop("image")
            
            # Update data in-place (Faster than |= for small dicts)
            data.update(augmented)
            
            # Optimized Type Casting
            for k, v in data.items():
                # Converts non‑float tensors/arrays to float32
                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                    data[k] = v.to(torch.float32)
                elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                    data[k] = v.astype(np.float32)
                    
        return {**data, "meta": meta}
    
    # --- Properties ---
    @property
    def transform(self) -> A.Compose:
        """Return the transformation operations."""
        return self._transform
    
    @transform.setter
    def transform(self, value: Any = None):
        """Set the transformation operations.
        
        Args:
            value: Transformations to apply.
            
        Raises:
            TypeError: If ``transform`` is not None or an instance of
                albumentations.Compose.
        """
        if value is None:
            self._transform = None
            return
        
        if isinstance(value, (dict, box.Box)):
            value = A.Compose(**value)
        if not isinstance(value, A.Compose):
            raise TypeError(f"Expected 'transform' to be either None, a dict/box.Box, "
                            f"or an instance of albumentations.Compose, but got "
                            f"{type(value).__name__}.")
        
        # Add additional targets to A.Compose if needed.
        existing_targets = value.processors.get("additional_targets", {})
        new_targets = {
            k: v.type for k, v in self._modalities.items()
            if v.type and v.module and k not in A.TARGET_TYPES and k not in existing_targets
        }
        
        if new_targets:
            value.add_targets(additional_targets=new_targets)
        
        self._transform = value
    
    # --- Data Loading ---
    def verify(self):
        """Verify dataset integrity.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset!")
        
        pk, _ = self.primary_modality
        for k, v in self._datapoints.items():
            if k not in self._modalities:
                raise RuntimeError(f"Expected 'datapoints' to have only defined "
                                   f"modalities, but got unexpected key: {k}")
            if self._modalities[k]:
                if v in [None, []]:
                    raise RuntimeError(f"Datapoint modality ``{k}`` is empty!")
                elif len(v) != self.__len__():
                    raise RuntimeError(f"Datapoint modality ``{k}`` has inconsistent"
                                       f"length with the dataset: {len(v)} != {self.__len__()}.")
        
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    # --- Access ---
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.
        
        Args:
            index: Index of datapoint.
            
        Returns:
            A dictionary containing all modalities for the specified datapoint.
        """
        # Efficiency: Use dict comprehension for faster construction
        return {
            k: (v[index] if v is not None else None)
            for k, v in self._datapoints.items()
        }


# ==============================================================================
# LOADERS
# ==============================================================================

class ImageLoader(ImageDataset):
    """A concrete class for image-only datasets.
    
    Extend ``ImageDataset`` and is designed to load images from a specified
    ``root``. The ``root`` can be a single image file, a directory, or a glob
    pattern. This class is primarily used for inference/prediction pipelines
    where no ground-truth labels are available.
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
            root: Root path to load images from. Can be a file, directory, or
                glob pattern.
            split: Data split subset to use. One of: Split.TRAIN, Split.VAL,
                Split.TEST, or Split.PREDICT. Defaults to Split.TRAIN.
            transform: Transformations to apply. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ``ClassList`` instance. If given, this will override any
                ``classes`` defined in the subclass. Defaults to None.
            verbose: If True, enables verbose output. Defaults to True.
        """
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            classlist = classlist,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # --- Data Loading ----
    def _load_primary_data(self) -> list:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of primary modality data files.
            
        Raises:
            IOError: If the ``root`` path is invalid.
        """
        root = self._root
        
        if root.is_image_file():
            paths = [root]
        elif "*" in str(root):
            # Using iglob (iterator) is more memory efficient than glob.glob
            paths = [Path(p) for p in glob.iglob(str(root), recursive=True)]
        elif root.is_dir() and root.exists():
            paths = list(root.rglob("*"))
        else:
            raise IOError(f"Invalid 'root' path: {root}")
        
        if not paths:
            return []
        
        images: list[Image] = []
        disable_pbar = self.disable_pbar
        split_str    = self.split_str
        
        with create_progress_bar(disable=disable_pbar) as pbar:
            paths = sorted(paths)
            desc  = f"Listing {self.__class__.__name__} {split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=root))
        
        return images


# ==============================================================================
# UTILITIES
# ==============================================================================

# --- Validation & Sanitization ---
