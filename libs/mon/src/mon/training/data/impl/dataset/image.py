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

from mon.core import create_progress_bar, log, Path, Split, Task
from mon.core.dtypes import ClassList, Image
from mon.training.augment import albumentations as A
from ...base import Dataset, Modalities, Modality
from ...comp import BatchCollateMixin, MultimodalDataLoadMixin, RegistrableMixin


# ==============================================================================
# DATASETS
# ==============================================================================

class ImageDataset(
    Dataset,
    RegistrableMixin,
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
        _tasks (list[Task]): A list of supported tasks. Defaults to an empty
            list and should be overridden in subclasses.
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
    _tasks     : list[Task]  = []
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
        if not self.modalities:
            raise ValueError("``modalities`` has no defined attributes.")
        
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
        """Return the string representation of the dataset."""
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            lines += [f"Root location: {self.root}"]
        if hasattr(self, "transform") and self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)
    
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the dataset (i.e., number of datapoints)."""
        pk, _ = self.primary_modality
        return len(self.datapoints[pk])
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the datapoint at the specified ``index`` in ``_datapoints``.
        
        Args:
            index: Index of datapoint.
            
        Returns:
            A dictionary containing the datapoint and its metadata.
        """
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.
        
        if self._transform:
            pk, _          = self.primary_modality
            args           = {k: v for k, v in data.items() if v is not None}
            args["image"]  = args.pop(pk)
            augmented      = self._transform(**args)
            augmented[pk]  = augmented.pop("image")
            data          |= augmented
            # Post-processing
            for k, v in data.items():
                # Converts non‑float tensors/arrays to float32
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
    def transform(self, transform: Any = None):
        """Setter for transformation operations.
        
        Args:
            transform: Transformations to apply.
            
        Raises:
            TypeError: If ``transform`` is not None or an instance of
                albumentations.Compose.
        """
        if isinstance(transform, dict | box.Box):
            transform = A.Compose(**transform)
        if transform is not None and not isinstance(transform, A.Compose):
            raise TypeError(f"``transform`` must be None or an instance of "
                            f"albumentations.Compose, got: {type(transform)}.")
        
        # Add additional targets to A.Compose if needed.
        if transform:
            additional_targets = {}
            # Adds modality‑specific transform targets if needed
            for k, v in self.modalities.items():
                if v.type is None or v.module is None:
                    continue
                if (k not in A.TARGET_TYPES and
                    k not in transform.additional_targets):
                    additional_targets[k] = v.type
            if len(additional_targets) > 0:
                transform.add_targets(additional_targets=additional_targets)
        
        self._transform = transform
    
    # --- Data Loading ---
    def verify(self):
        """Verify dataset integrity.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset!")
        
        pk, _ = self.primary_modality
        for k, v in self.datapoints.items():
            if k not in self.modalities:
                raise RuntimeError(f"Modality ``{k}`` is not defined in ``modalities``. "
                                   f"Define it in the class if intentional.")
            if self.modalities[k]:
                if v in [None, []]:
                    raise RuntimeError(f"``datapoints`` has no ``{k}`` attributes!")
                elif len(v) != self.__len__():
                    raise RuntimeError(f"Number of ``{k}`` items does not match number "
                                       f"of ``{pk}``, got: {len(v)} != {self.__len__()}")
        
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
        datapoint = {}
        for k, v in self.datapoints.items():
            if v is not None:
                datapoint[k] = v[index]
            else:
                datapoint[k] = None
        return datapoint


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
        root = self.root
        if root.is_image_file():
            paths = [root]
        elif root.is_dir() and root.exists():
            paths = list(root.rglob("*"))
        elif "*" in str(root):
            paths = [Path(i) for i in glob.glob(str(root))]
        else:
            raise IOError(f"Invalid root path: {root}")
        
        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(paths)
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=root))
        
        return images


# ==============================================================================
# UTILITIES
# ==============================================================================

# --- Validation & Sanitization ---
