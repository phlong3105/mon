#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluation datasets.

This module provides dataset classes specifically designed for evaluation
purposes, such as image quality assessment (IQA). These datasets support
loading input and target data, applying transformations, and preparing data
for evaluation pipelines outside the standard training/evaluation/testing
loops.
"""

__all__ = [
    "ImageEvalDataset",
]

from typing import Any, Optional

import box
import numpy as np
import torch

from mon.core import create_progress_bar, log, Path
from mon.core.dtypes import ClassList, Image
from mon.training.augment import albumentations as A
from ...base import Dataset
from ...comp import BatchCollateMixin, InputTargetLoadMixin


# ==============================================================================
# DATASETS
# ==============================================================================

class ImageEvalDataset(Dataset, InputTargetLoadMixin, BatchCollateMixin):
    """A concrete class for image quality assessment (IQA) datasets.
    
    Define two main modalities: ``image`` and ``target`` (also an image, optional).
    Primarily used for separated evaluation pipelines outside the train/eval/test
    loop.
    
    Attributes:
        _transform (albumentations.Compose): Transformations for input/target.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir : Path,
        target_dir: Path             = None,
        transform : A.Compose        = None,
        classlist : Path | ClassList = None,
        verbose   : bool             = True,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            input_dir: Absolute path to the input/predict data directory.
            target_dir: Absolute path to the target directory. Defaults to None.
            transform: Transformations to apply to input/target. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ``ClassList`` instance. If given, this will override any
                ``classes`` defined in the subclass. Defaults to None.
            verbose: If True, enables verbose output. Defaults to True.
        """
        super().__init__(
            input_dir  = input_dir,
            target_dir = target_dir,
            classlist  = classlist,
            verbose    = verbose,
            *args, **kwargs
        )
        self.transform  = transform
    
    def __del__(self):
        """Close the dataset loading mechanism and releases resources."""
        pass
    
    # --- Representation ---
    def __repr__(self) -> str:
        """Return the string representation of the dataset."""
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        if hasattr(self, "transform") and self._transform:
            lines += [repr(self._transform)]
        return "\n".join(lines)
    
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the dataset (i.e., number of datapoints)."""
        return len(self.datapoints["image"])
    
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
            if self.has_target:
                augmented      = self.transform(image=data["image"], target=data["target"])
                data["image"]  = augmented["image"]
                data["target"] = augmented["target"]
            else:
                augmented      = self.transform(image=data["image"])
                data["image"]  = augmented["image"]
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
        
        # Add additional targets to A.Compose if needed.
        if transform and self.has_target:
            transform.add_targets(additional_targets={"target": "image"})
        
        self._transform = transform
        
    # --- Data Loading ---
    def _load_data(self) -> dict[str, Any]:
        """Core data loading mechanism for the dataset."""
        # Initialize empty datapoints dictionary with modalities
        datapoints = {}
        
        # List image
        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(self.input_dir.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=self.input_dir))
        datapoints["image"] = images
        
        # List target
        targets: Optional[list[Image]] = None
        if self.has_target:
            targets: list[Image] = []
            with create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} target image(s)"
                for image in pbar.track(sequence=images, description=desc):
                    target_file = self.target_dir / image.path.name
                    target_file = target_file.image_file(exist=True)
                    if target_file.is_image_file():
                        targets.append(Image(data=target_file, root=self.target_dir))
        datapoints["target"] = targets
        
        # List metadata
        datapoints["meta"] = [i.meta for i in images]
        
        return datapoints
    
    def verify(self):
        """Verify dataset integrity.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset!")
        
        for k, v in self.datapoints.items():
            if v in [None, []]:
                raise RuntimeError(f"``datapoints`` has no ``{k}`` attributes!")
            elif len(v) != self.__len__():
                raise RuntimeError(f"Number of ``{k}`` items does not match number "
                                   f"of ``image``, got: {len(v)} != {self.__len__()}")
        
        if self.verbose:
            log(f"Number of datapoints: {self.__len__()}.")
    
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
# UTILITIES
# ==============================================================================

# --- Validation & Sanitization ---
