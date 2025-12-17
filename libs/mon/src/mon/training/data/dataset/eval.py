#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for evaluation datasets.

This module provides dataset classes specifically designed for evaluation
purposes, such as image quality assessment (IQA). These datasets support
loading input and target data, applying transformations, and preparing data
for evaluation pipelines outside the standard training/evaluation/testing
loops.
"""

__all__ = [
    "ImageEvalDataset",
]

from typing import Any

import box
import numpy as np
import torch

from mon.core import create_progress_bar, log, Path
from mon.core.dtypes import Image
from mon.training.augment import albumentations as A
from .base import DataLoaderMixin, Dataset, DatasetLoadingMixin
from ..classes import Classes


class ImageEvalDataset(Dataset, DatasetLoadingMixin, DataLoaderMixin):
    """A concrete class for evaluation datasets.
    
    This class defines a concrete implementation of image quality assessment (IQA)
    datasets. It includes two main modalities: ``image`` and ``target`` (optional).
    This is primarily used for separated evaluation pipelines outside the
    train/eval/test loop.
    
    Attributes:
        input_dir (Path): Absolute path to the input/predict data directory.
        target_dir (Path): Absolute path to the target data directory. Defaults
            to None means no target data.
        transform (albumentations.Compose): Transformations for input/target.
    """
    
    def __init__(
        self,
        input_dir : Path,
        target_dir: Path           = None,
        transform : A.Compose      = None,
        classes   : Path | Classes = None,
        verbose   : bool           = True,
        *args, **kwargs
    ):
        """Initializes the ImageEvalDataset instance.
        
        Args:
            input_dir (Path): Absolute path to the input/predict data directory.
            target_dir (Path, optional): Absolute path to the target data directory.
                Defaults to None.
            transform (albumentations.Compose, optional): Transformations for
                input/target. Defaults to None.
            classes (Path or Classes, optional): Path to classes file or Classes
                instance. Defaults to None.
            verbose (bool): If True, enables verbose output. Defaults to True.
        """
        self.input_dir  = input_dir
        self.target_dir = target_dir
        self.transform  = transform
        super().__init__(classes=classes, verbose=verbose, *args, **kwargs)
        
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
        
        if self.transform:
            if self.has_target:
                augmented = self.transform(image=data["image"], target=data["target"])
                data["image"]  = augmented["image"]
                data["target"] = augmented["target"]
            else:
                augmented = self.transform(image=data["image"])
                data["image"] = augmented["image"]
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
            int: The number of datapoints in the dataset.
        """
        return len(self.datapoints["image"])
    
    def __repr__(self) -> str:
        """Returns the next datapoint in the dataset iteration.
        
        Returns:
            dict[str, Any]: A dictionary containing the next datapoint.
        
        Raises:
            StopIteration: If index exceeds the dataset length.
        """
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        if hasattr(self, "transform") and self._transform:
            lines += [repr(self._transform)]
        return "\n".join(lines)
    
    # ----- Properties -----
    @property
    def input_dir(self) -> Path:
        """Getter for the input/predict data directory.
        
        Returns:
            Path: Path to the input/predict data directory.
        """
        return self._input_dir
    
    @input_dir.setter
    def input_dir(self, input_dir: Path):
        """Setter for the input/predict data directory.
        
        Args:
            input_dir (Path): Path to the input/predict data directory.
            
        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"``input_dir`` directory not found: {input_dir}.")
        self._input_dir = input_dir
    
    @property
    def target_dir(self) -> Path:
        """Getter for the target data directory.
        
        Returns:
            Path: Path to the target data directory.
        """
        return self._target_dir
    
    @target_dir.setter
    def target_dir(self, target_dir: Path = None):
        """Setter for the target data directory.
        
        Args:
            target_dir (Path, optional): Path to the target data directory.
                Defaults to None.
                
        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        if target_dir is not None:
            target_dir = Path(target_dir)
            if not target_dir.is_dir():
                raise FileNotFoundError(f"``target_dir`` directory not found: {target_dir}.")
        self._target_dir = target_dir
    
    @property
    def has_target(self) -> bool:
        """Indicates whether the dataset has target data.
        
        Returns:
            bool: True if target data is available, False otherwise.
        """
        return self.target_dir is not None and self.target_dir.is_dir()
    
    @property
    def transform(self) -> A.Compose:
        """Getter for transformation operations.
        
        Returns:
            albumentations.Compose: Transformations for input/target.
        """
        return self._transform
    
    @transform.setter
    def transform(self, transform: Any):
        """Setter for transformation operations.
        
        Args:
            transform (Any): Transformations for input/target.
            
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
        if transform and self.has_target:
            transform.add_targets(additional_targets={"target": "image"})
        
        self._transform = transform
        
    # ----- Initialize -----
    def _core_load(self) -> dict[str, Any]:
        """Core loading mechanism for the dataset."""
        datapoints = {}
        
        # Image
        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(self.input_dir.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=self.input_dir))
        datapoints["image"]  = images
        
        # Target
        targets: list[Image] = None
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
        
        return datapoints
      
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
        """Verifies dataset integrity after loading.
        
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
    
    # ----- Access -----
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Gets a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint.
        """
        datapoint = {}
        for k, v in self._datapoints.items():
            if v is not None and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
            else:
                datapoint[k] = None
        return datapoint
    
    def _get_meta(self, index: int) -> dict[str, Any]:
        """Gets metadata at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the metadata.
        """
        return self._datapoints["image"][index].meta
    
    # ----- Utils -----
    def collate_fn(self, batch: list[dict]) -> dict[str, Any]:
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
            if v is None:
                zipped[k] = None
            elif isinstance(v[0], torch.Tensor):
                zipped[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], np.ndarray):
                zipped[k] = np.stack(v, axis=0)

        return zipped
