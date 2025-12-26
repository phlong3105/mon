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
import os
from typing import Any

import box
import numpy as np
import torch

from mon.core import create_progress_bar, log, Path, rich, Split, Task
from mon.core.dtypes import Image
from mon.training.augment import albumentations as A
from .base import (
    DataLoaderMixin,
    Dataset,
    DatasetMetadataMixin,
    DatasetMultimodalLoadingMixin,
    Modalities,
    Modality,
)
from ..classes import Classes


# --- Image Dataset ---
class ImageDataset(
    Dataset,
    DatasetMetadataMixin,
    DatasetMultimodalLoadingMixin,
    DataLoaderMixin
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
        _classes (Classes): The dataset classes/labels. Defaults to None and
            should be overridden in subclasses.
        transform (albumentations.Compose): Transformations to apply to the data.
        verbose (bool): If True, enables verbose output.
    """
    
    _subset    : str         = None
    _tasks     : list[Task]  = []
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classes   : Classes     = None
    
    def __init__(
        self,
        root     : Path,
        split    : Split          = Split.TRAIN,
        transform: A.Compose      = None,
        classes  : Path | Classes = None,
        verbose  : bool           = True,
        *args, **kwargs
    ):
        """Initializes the BaseDataset instance.
        
        Args:
            root (Path): Absolute path to the dataset root directory.
            split (Split): Data split subset to use. One of: Split.TRAIN,
                Split.VAL, Split.TEST, or Split.PREDICT. Defaults to Split.TRAIN.
            transform (albumentations.Compose, optional): Transformations to
                apply. Defaults to None.
            classes (Path or Classes, optional): Either a path to a .yaml file
                containing class label definitions, or a Classes instance.
                If given, this will override any classes defined in the
                subclass. Defaults to None.
            verbose (bool): If True, enables verbose output. Defaults to True.
        
        Raises:
            ValueError: If ``modalities`` has no defined attributes.
        """
        # Validate modalities
        if not self.modalities:
            raise ValueError("``modalities`` has no defined attributes.")
        
        super().__init__(
            root    = root,
            split   = split,
            classes = classes,
            verbose = verbose,
            *args, **kwargs
        )
        self.transform = transform
    
    # --- Magic Methods ---
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
    
    def __repr__(self) -> str:
        """Returns the string representation of the dataset.
        
        Returns:
            str: String representation of the dataset.
        """
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            lines += [f"Root location: {self.root}"]
        if hasattr(self, "transform") and self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)
    
    # --- Properties ---
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
    
    # --- Data Loading ----
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
                        images.append(Image(data=path, root=pattern))
        
        return images
    
    def _load_modality_data(self, key: str) -> list[Any]:
        """Loads modality data files in the dataset.
        
        Args:
            key (str): The modality key to load.
            
        Returns:
            list[Any]: A list of modality data files.
        """
        pk, pk_modality = self.primary_modality
        pk_name  = pk_modality.name
        pk_files = self.datapoints[pk]
        
        modality = self.modalities[key]
        name     = modality.name
        module   = modality.module
        files    = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for file in pbar.track(
                sequence    = pk_files,
                description = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            ):
                path = file.path.replace_part(f"{os.sep}{pk_name}{os.sep}", f"{os.sep}{name}{os.sep}")
                files.append(module(path=path, root=file.root))
        
        return files
    
    def _on_load_start(self):
        """A hook method called at the start of the data loading process.
        
        This method can be overridden by subclasses to perform additional
        operations before the dataset is loaded.
        """
        pass
    
    def _on_load_end(self):
        """A hook method called at the end of the data loading process.
        
        This method can be overridden by subclasses to perform additional
        operations after the dataset has been loaded.
        """
        pass
    
    def verify(self):
        """Verifies dataset integrity.
        
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
    
    # --- Data Retrieval ---
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
    
    # --- Utils ---
    def collate_fn(self, batch: list[dict]) -> dict:
        """Collates a batch of input items for torch.utils.data.dataset.DataLoader.
        
        By default, batch is a list of dicts, where each dict is a datapoint.
        We need to collate these into a single dict where each key corresponds to
        a modality and the values are stacked tensors or arrays.

        Args:
            batch (list[dict]): List of dicts, each dict is a datapoint.

        Returns:
            dict[str, Any]: Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        
        for k, v in zipped.items():
            if k not in self._modalities:  # i.e., metadata
                continue
            if v is None:
                zipped[k] = None
            elif isinstance(v[0], torch.Tensor):
                zipped[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], np.ndarray):
                zipped[k] = np.stack(v, axis=0)
        
        return zipped
    

# --- Image Loader ---
class ImageLoader(ImageDataset):
    """A concrete class for image-only datasets.
    
    This class extends ``ImageDataset`` and is designed to load images from a
    specified ``root``. The ``root`` can be a single image file, a directory,
    or a glob pattern. This class is primarily used for inference/prediction
    pipelines where no ground-truth labels are available.
    """
    
    def __init__(
        self,
        root     : Path,
        split    : Split          = Split.PREDICT,
        transform: A.Compose      = None,
        classes  : Path | Classes = None,
        verbose  : bool           = True,
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
    
    # --- Initialize ---
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
                    images.append(Image(data=path, root=root))
        
        return images
