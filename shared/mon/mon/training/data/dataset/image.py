#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image-based datasets.

This module provides base classes for image datasets and data loaders, including
functionality for loading images and applying transformations.
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

from mon.core import create_progress_bar, Path, rich, Split
from mon.core.dtypes import Image
from mon.training.augment import albumentations as A
from .base import BaseDataset, Modalities, Modality
from ..classes import Classes


# ----- Base Image Dataset -----
class ImageDataset(BaseDataset):
    """A base class for image-based datasets where images are the primary modality.
    It can be extended to accommodate additional modalities (e.g., depth maps,
    segmentation masks, bounding boxes, captions, or other sensor data).
    
    This class extends the BaseDataset class and provides functionality for
    loading and transforming image data. It implements a basic data loading and
    transformation logic for datasets where the image are located at:
    ``self.root/self.split_str/self.primary_modality.name/``.
    
    For other use cases, this class need to be extended by concrete implementations.
    For the most part, the ``modalities`` attributes must be defined to specify
    the modalities present in the dataset. In addition, the ``list_primary_data()``
    method must be implemented to list the primary modality data. The other
    modalities are assumed to be at the same locations with the primary modality
    and will be listed automatically.
    
    Attributes:
        _modalities (Modalities): A dictionary defining the dataset modalities.
        transform (albumentations.Compose): Transformations to apply to the data.
    """
    
    _modalities: Modalities = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    
    # ----- Magic Methods -----
    def __del__(self):
        """Closes the dataset loading mechanism and releases resources."""
        pass
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Gets a datapoint and metadata at the specified ``index``.
        
        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint and metadata.
        """
        data = self._get_datapoint(index=index)
        meta = self._get_meta(index=index)
        
        if self._transform:
            pk, _         = self.primary_modality
            args          = {k: v for k, v in data.items() if v is not None}
            args["image"] = args.pop(pk)
            augmented     = self._transform(**args)
            augmented[pk] = augmented.pop("image")
            data     |= augmented
            # Convert to float32 if necessary
            for k, v in data.items():
                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                    data[k] = v.to(torch.float32)
                elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                    data[k] = v.astype(np.float32)
                    
        return data | {"meta": meta}
    
    def __len__(self) -> int:
        """Returns the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        pk, _ = self.primary_modality
        return len(self._datapoints[pk])
    
    # ----- Properties -----
    @property
    def transform(self) -> A.Compose:
        """Getter for transformation operations.
        
        Returns:
            albumentations.Compose: Transformations to apply.
        """
        return self._transform
    
    @transform.setter
    def transform(self, transform: Any = None):
        """Setter for transformation operations.
        
        Args:
            transform (Any): Transformations to apply.
            
        Raises:
            TypeError: If ``transform`` is not None or an instance of
                albumentations.Compose.
        """
        if isinstance(transform, dict | box.Box):
            transform = A.build_compose(**transform)
        if transform is not None and not isinstance(transform, A.Compose):
            raise TypeError(f"``transform`` must be None or an instance of "
                            f"albumentations.Compose, got: {type(transform)}.")
        
        # Add additional targets to A.Compose if needed.
        if transform:
            additional_targets = {}
            for k, v in self.modalities.items():
                if v.type is None or v.module is None:
                    continue
                if (k not in A.TARGET_TYPES and
                    k not in transform.additional_targets):
                    additional_targets[k] = v.type
            if len(additional_targets) > 0:
                transform.add_targets(additional_targets=additional_targets)
        
        self._transform = transform
    
    # ----- Initialize -----
    def _load_primary_data(self) -> list[Any]:
        """Loads primary modality data files in the dataset.
        
        Returns:
            list[Any]: A list of primary modality data files.
        """
        pk, pk_modality = self.primary_modality
        pk_name  = pk_modality.name
        
        patterns = [self._root / self.split_str / pk_name]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
        
        return images
    
    # ----- Data Retrieval -----
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Gets a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint.
        """
        datapoint = {}
        for k, v in self._datapoints.items():
            if hasattr(v[index], "data"):
                datapoint[k] = v[index].data
            else:
                datapoint[k] = None
        return datapoint
    
    def _get_meta(self, index: int) -> dict:
        """Gets metadata at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the metadata.
        """
        pk, _ = self.primary_modality
        return self._datapoints[pk][index].meta


# ----- Image Loader -----
class ImageLoader(ImageDataset):
    """A convenient dataset that loads only images from a file path, pattern,
    or directory.
    
    This class extends the ImageDataset class and is designed to load images
    from a specified root path. It can handle single image files, directories,
    or glob patterns.
    
    This is primarily used for inference/prediction pipelines (i.e., no ground-truth).
    """
    
    def __init__(
        self,
        root     : Path,
        split    : Split        = Split.PREDICT,
        transform: A.Compose    = None,
        classes: Path | Classes = None,
        verbose  : bool         = True,
        *args, **kwargs
    ):
        """Initializes the ImageLoader dataset.
        
        Args:
            root (Path): Root path to load images from. Can be a file, directory,
                or glob pattern.
            split (Split): Dataset split type. Default is Split.PREDICT.
            transform (albumentations.Compose): Transformations to apply to the
                images.
            classes (Path or Classes): Path to classes file or Classes object.
            verbose (bool): Whether to print dataset information. Default is True.
        """
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            classes   = classes,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # ----- Initialize -----
    def _load_primary_data(self) -> list:
        """Loads primary modality data files in the dataset.
        
        Returns:
            list[Any]: A list of primary modality data files.
            
        Raises:
            IOError: If the root path is invalid.
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
                    images.append(Image(path=path, root=root))
        
        return images
