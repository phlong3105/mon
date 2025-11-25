#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements dataset classes where image data is the primary modality.

Note: we use the ``albumentations`` library for transformations and augmentations.
"""

__all__ = [
    "ImageDataset",
    "ImageDualDomainDataset",
    "ImageEvalDataset",
    "ImageLoader",
]

import abc
import glob
import random

import box
import numpy as np
import torch

from mon.core import create_progress_bar, Path, rich, Split
from mon.core.dtypes import Image
from mon.training.augment import albumentations as A
from .base import (
    BaseDataset,
    BaseDualDomainDataset,
    BaseEvalDataset,
    Modalities,
    Modality,
)


# ----- Image Dataset -----
class ImageDataset(BaseDataset):
    """A general class for image-based datasets where images are the primary modality.
    It can be extended to accommodate additional modalities (e.g., depth maps,
    segmentation masks, bounding boxes, captions, or other sensor data).
    
    This class implements a basic data loading and transformation logic for datasets
    where the image are located at: ``self.root/self.split_str/self.primary_modality.name/``.
    
    For other use cases, this class need to be extended by concrete implementations.
    For the most part, the ``modalities`` attributes must be defined to specify
    the modalities present in the dataset. In addition, the ``list_primary_data()``
    method must be implemented to list the primary modality data. The other
    modalities are assumed to be at the same locations with the primary modality
    and will be listed automatically.
    
    Attributes:
        modalities: Dictionary of datapoint modalities.
        
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.TRAIN``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    modalities: Modalities = {
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
                if (k not in A.TARGET_TYPES and
                    k not in self.transform.additional_targets):
                    additional_targets[k] = v.type
            if len(additional_targets) > 0:
                self.transform.add_targets(additional_targets=additional_targets)
    
    def list_primary_data(self) -> list:
        """Lists ``datapoints`` with image annotations for split."""
        pk, pk_modality = self.primary_modality
        pk_name  = pk_modality.name
        
        patterns = [self.root / self.split_str / pk_name]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
    
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


class ImageDualDomainDataset(BaseDualDomainDataset, abc.ABC):
    """A base class for dual-domain vision datasets where images/frames are the
    primary modality. It can be extended to accommodate additional modalities
    (e.g., depth maps, segmentation masks, bounding boxes, captions, or other
    sensor data).
    
    It is mainly used in Image-to-Image translation tasks. It requires two directories
    to host data from two domains A and B. The number of items in each directory
    can be the same (paired) or different (unpaired/unaligned).
    
    **This is still a work in progress, so I will update the documentation later.**
    
    Attributes:
        modalities_A: Dictionary of datapoint modalities in domain A.
        modalities_B: Dictionary of datapoint modalities in domain B.
        
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.TRAIN``.
        transform: Transformations for input/target. Default: ``None``.
        serial: If ``True``, select images in domain B sequentially.
            Otherwise, randomly select images. Default: ``False``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    modalities_A: Modalities = {
        "image_A": Modality(name="image_A", type="image", module=Image, train=True, test=True, primary=True),
    }
    modalities_B: Modalities = {
        "image_B": Modality(name="image_B", type="image", module=Image, train=True, test=True, primary=True),
    }
    
    def __init__(
        self,
        root     : Path,
        split    : Split     = Split.TRAIN,
        transform: A.Compose = None,
        serial   : bool      = False,
        verbose  : bool      = True,
        *args, **kwargs
    ):
        self.serial = serial
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            verbose   = verbose,
            *args, **kwargs
        )
    
    # ----- Magic Methods -----
    def __getitem__(self, index: int) -> dict:
        """Retrieves a datapoint and metadata at given ``index`` as a ``dict``.
        Here, we assume domain A is the primary.
        """
        index_A = index % self.size(domain="A")
        if self.serial:
            index_B = index % self.size(domain="B")
        else:
            index_B = random.randint(0, self.size(domain="B") - 1)
        
        datapoint_A = self.get_datapoint(domain="A", index=index_A)
        datapoint_B = self.get_datapoint(domain="B", index=index_B)
        datapoint   = datapoint_A | datapoint_B
        meta_A      = self.get_meta(domain="A", index=index_A)
        meta_B      = self.get_meta(domain="B", index=index_B)
        
        if self.transform:
            pk, _          = self.primary_modality(domain="A")
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
                    
        return datapoint | {
            "meta_A": meta_A,
            "meta_B": meta_B,
        }
    
    def __len__(self) -> int:
        """Retrieves the total number of datapoints.
        
        As we have two datasets with potentially different numbers of images,
        we take a maximum of.
        """
        return max(self.size(domain="A"), self.size(domain="B"))
    
    # ----- Properties -----
    def size(self, domain: str) -> int:
        """Returns the number of items in the ``domain``."""
        datapoints = self.datapoints_A if domain == "A" else self.datapoints_B
        pk, _      = self.primary_modality(domain=domain)
        return len(datapoints[pk])
    
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
            # Domain A
            for k_A, v_A in self.modalities_A.items():
                if v_A.type is None or v_A.module is None:
                    continue
                if (k_A not in A.TARGET_TYPES and
                    k_A not in self.transform.additional_targets):
                    additional_targets[k_A] = v_A.type
            # Domain B
            for k_B, v_B in self.modalities_B.items():
                if v_B.type is None or v_B.module is None:
                    continue
                if (k_B not in A.TARGET_TYPES and
                    k_B not in self.transform.additional_targets):
                    additional_targets[k_B] = v_B.type
            if len(additional_targets) > 0:
                self.transform.add_targets(additional_targets=additional_targets)
    
    def reset(self):
        """Resets the dataset to start over."""
        self.index = 0
    
    def close(self):
        """Closes and releases dataset resources."""
        pass
    
    # ----- Data Retrieval -----
    def get_datapoint(self, domain: str, index: int) -> dict:
        """Gets a datapoint in the ``domain`` at the specified ``index``.

        Args:
            domain: Domain name.
            index: Index of datapoint.

        Returns:
            A ``dict`` containing the datapoint.
        """
        datapoints = self.datapoints_A if domain == "A" else self.datapoints_B
        datapoint  = {}
        for k, v in datapoints.items():
            if hasattr(v[index], "data"):
                datapoint[k] = v[index].data
            else:
                datapoint[k] = None
        return datapoint
    
    def get_meta(self, domain: str, index: int) -> dict:
        """Gets metadata in the ``domain`` at the specified ``index``.

        Args:
            domain: Domain name.
            index: Index of metadata.
            
        Returns:
            A ``dict`` containing the metadata.
        """
        datapoints = self.datapoints_A if domain == "A" else self.datapoints_B
        pk, _      = self.primary_modality(domain=domain)
        return datapoints[pk][index].meta


# ----- Image Loader -----
class ImageLoader(ImageDataset):
    """A convenient dataset that loads images from a file path, pattern, or directory.
    
    This is primarily used for inference/prediction on images stored on disk
    (i.e., without ground-truth).
    
    Attributes:
        modalities: Dictionary of datapoint modalities.
        
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.PREDICT``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    modalities: Modalities = {
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
class ImageEvalDataset(BaseEvalDataset):
    """"""
    
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
