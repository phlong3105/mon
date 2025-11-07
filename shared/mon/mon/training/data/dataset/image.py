#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements dataset classes where image data is the primary modality."""

__all__ = [
    "ImageEvalDataset",
    "ImageLoader",
]

import glob

import box
import numpy as np
import torch

from mon.core import create_progress_bar, Path, Split
from mon.core.dtypes import Image
from mon.training import albumentations as A
from .base import EvalDataset, Modalities, Modality
from .vision import VisionDataset


# ----- Image Loader -----
class ImageLoader(VisionDataset):
    """Loads images from a file path, pattern, or directory.
    
    Attributes:
        modalities: Dictionary of datapoint modalities.
        
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.PREDICT``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    
    def __init__(
        self,
        root     : Path,
        split    : Split     = Split.PREDICT,
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
    
    # ----- Initialize -----
    def list_primary_data(self) -> list:
        """Retrieves ``Image`` objects from the root path.

        Raises:
            IOError: If ``root`` path invalid or no images found.
        """
        if self.root.is_image_file():
            paths = [self.root]
        elif self.root.is_dir() and self.root.exists():
            paths = list(self.root.rglob("*"))
        elif "*" in str(self.root):
            paths = [Path(i) for i in glob.glob(str(self.root))]
        else:
            raise IOError(f"Invalid root path: {self.root}")
        
        images: list[Image] = []
        with create_progress_bar() as pbar:
            paths = sorted(paths)
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(path=path, root=self.root))
        
        return images


# ----- Evaluation Dataset -----
class ImageEvalDataset(EvalDataset):
    
    # ----- Magic Methods -----
    def __getitem__(self, index: int) -> dict:
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        
        if self.transform:
            if self.has_target:
                augmented = self.transform(image=datapoint["image"], target=datapoint["target"])
                datapoint["image"]  = augmented["image"]
                datapoint["target"] = augmented["target"]
            else:
                augmented = self.transform(image=datapoint["image"])
                datapoint["image"] = augmented["image"]
            # Convert to float32 if necessary
            for k, v in datapoint.items():
                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                    datapoint[k] = v.to(torch.float32)
                elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                    datapoint[k] = v.astype(np.float32)
                    
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        return len(self.datapoints["image"])
    
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
        if self.transform and self.has_target:
            transform.add_targets(additional_targets={"target": "image"})
        
    def init_data(self):
        """Initializes all datapoints in the dataset."""
        # Image
        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(self.input_dir.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(path=path, root=self.input_dir))
        
        # Target
        targets: list[Image] = None
        if self.has_target:
            targets: list[Image] = []
            with create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} target image(s)"
                for file in pbar.track(sequence=images, description=desc):
                    path = self.target_dir / file.path.name
                    path = path.image_file()
                    if path.is_image_file():
                        targets.append(Image(path=path, root=self.target_dir))
        
        # Initialize datapoints
        self.datapoints["image"]  = images
        self.datapoints["target"] = targets
        
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
            if v is not None and hasattr(v[index], "data"):
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
        return self.datapoints["image"][index].meta
