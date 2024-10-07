#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Dataset Template.

This module implements base classes for all datasets.
"""

from __future__ import annotations

__all__ = [
    "ChainDataset",
    "ConcatDataset",
    "Dataset",
    "IterableDataset",
    "MultimodalDataset",
    "Subset",
    "TensorDataset",
    "random_split",
]

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon import core
from mon.core import pathlib
from mon.core.data import annotation
from mon.core.rich import console
from mon.core.transform import albumentation as A
from mon.globals import DEPTH_DATA_SOURCES, Split, Task

ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
DepthMapAnnotation  = annotation.DepthMapAnnotation
ImageAnnotation     = annotation.ImageAnnotation


# region Base

class Dataset(dataset.Dataset, ABC):
    """The base class of all datasets.
    
    Attributes:
        tasks: A :obj:`list` of tasks that the dataset supports.
        splits: A :obj:`list` of splits that the dataset supports.
        has_test_annotations: If ``True``, the test set has ground-truth labels.
            Default: ``False``.
        datapoint_attrs: A :obj:`dict` of datapoint attributes with the keys
            are the attribute names and the values are the attribute types.
        classlabels: A :obj:`ClassLabels`, i.e., a list of class labels that the
            dataset supports. Default: ``None``.
        
    Args:
        root: The root directory where the data is stored.
        split: The data split to use. Default: ``'Split.TRAIN'``.
        transform: Transformations performed on both the input and target. We use
            `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
        to_tensor: If ``True``, convert input and target to :obj:`torch.Tensor`.
            Default: ``False``.
        cache_data: If ``True``, cache data to disk for faster loading next
            time. Default: ``False``.
        verbose: Verbosity. Default: ``True``.
    """
    
    tasks : list[Task]  = []
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    datapoint_attrs     = DatapointAttributes({})
    has_test_annotations: bool        = False
    classlabels         : ClassLabels = None
    
    def __init__(
        self,
        root      : pathlib.Path,
        split     : Split           = Split.TRAIN,
        transform : A.Compose | Any = None,
        to_tensor : bool            = False,
        cache_data: bool            = False,
        verbose   : bool            = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root       = pathlib.Path(root)
        self.split 	    = split
        self.transform  = transform
        self.to_tensor  = to_tensor
        self.verbose    = verbose
        self.index		= 0  # Use with :obj:`__iter__` and :meth`__next__`
        self.datapoints = {}
        self.init_transform()
        self.init_datapoints()
        self.init_data(cache_data=cache_data)
        
    # region Magic Methods
    
    def __del__(self):
        self.close()
    
    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Returns a dictionary containing the datapoint and metadata at the
        given :obj:`index`.
        """
        pass
    
    def __iter__(self):
        """Returns an iterator starting at the index ``0``."""
        self.reset()
        return self
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of datapoints in the dataset."""
        pass
    
    def __next__(self) -> dict:
        """Returns the next datapoint and metadata when using :obj:`__iter__`.
        """
        if self.index >= self.__len__():
            raise StopIteration
        else:
            result      = self.__getitem__(self.index)
            self.index += 1
            return result
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform:
            body += [repr(self.transform)]
        lines = [head]  # + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    # endregion
    
    # region Properties
    
    @property
    def disable_pbar(self) -> bool:
        return not self.verbose
    
    @property
    def has_annotations(self) -> bool:
        """Returns ``True`` if the images has accompanied annotations,
        otherwise ``False``.
        """
        return (
            (
                self.has_test_annotations
                and self.split in [Split.TEST, Split.PREDICT]
            )
            or (self.split in [Split.TRAIN, Split.VAL])
        )
    
    @property
    def hash(self) -> int:
        """Return the total hash value of all the files (if it has one).
        Hash values are integers (in bytes) of all files.
        """
        sum = 0
        for k, v in self.datapoints.items():
            if isinstance(v, list):
                for a in v:
                    if a and hasattr(a, "meta"):
                        sum += a.meta.get("hash", 0)
        return sum
    
    @property
    def main_attribute(self) -> str:
        """Return the main attribute of the dataset as the first key in
        :obj:`datapoint_attrs`.
        """
        return next(iter(self.datapoint_attrs.keys()))
    
    @property
    def new_datapoint(self) -> dict:
        """Return a new datapoint with default values."""
        return {k: None for k in self.datapoint_attrs.keys()}
    
    @property
    def split(self) -> Split:
        return self._split
    
    @split.setter
    def split(self, split: Split):
        split = Split[split] if isinstance(split, str) else split
        if split in self.splits:
            self._split = split
        else:
            raise ValueError(f"`split` must be one of {self.splits}, "
                             f"but got {split}.")
    
    @property
    def split_str(self) -> str:
        return self.split.value
    
    # endregion
    
    # region Initialization
    
    def init_transform(self, transform: A.Compose | Any = None):
        """Initialize transformation operations."""
        self.transform = transform or self.transform
    
    def init_datapoints(self):
        """Initialize datapoints dictionary."""
        if not self.datapoint_attrs:
            raise ValueError(f"`datapoint_attrs` has no defined attributes.")
        self.datapoints = {k: list[v]() for k, v in self.datapoint_attrs.items()}
    
    def init_data(self, cache_data: bool = False):
        """Initialize data."""
        # Get image from disk or cache
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.get_data()
            
        # Filter and verify data
        self.filter_data()
        self.verify_data()
        
        # Cache data
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            pathlib.delete_cache(cache_file)
    
    @abstractmethod
    def get_data(self):
        """Get the base data."""
        pass
    
    def cache_data(self, path: pathlib.Path):
        """Cache data to :obj:`path`."""
        hash_ = 0
        if path.is_cache_file():
            cache = torch.load(path)
            hash_ = cache.get("hash", 0)
        
        if self.hash != hash_:
            cache = self.datapoints | {"hash": self.hash}
            torch.save(cache, str(path))
            if self.verbose:
                console.log(f"Cached data to: {path}")
    
    def load_cache(self, path: pathlib.Path):
        """Load cache data from :obj:`path`."""
        self.datapoints = torch.load(path)
        self.datapoints.pop("hash", None)
    
    @abstractmethod
    def filter_data(self):
        """Filter unwanted datapoints."""
        pass
    
    @abstractmethod
    def verify_data(self):
        """Verify dataset."""
        pass
    
    @abstractmethod
    def reset(self):
        """Resets and starts over."""
        pass
    
    @abstractmethod
    def close(self):
        """Stops and releases."""
        pass
    
    # endregion
    
    # region Retrieve Data
    
    @abstractmethod
    def get_datapoint(self, index: int) -> dict:
        """Get a datapoint at the given :obj:`index`."""
        pass
    
    @abstractmethod
    def get_meta(self, index: int) -> dict:
        """Get metadata at the given :obj:`index`."""
        pass
    
    @classmethod
    def collate_fn(cls, batch: list[dict]) -> dict:
        """Collate function used to fused input items together when using
		:obj:`batch_size` > ``1``. This is used in
		:obj:`torch.utils.data.DataLoader` wrapper.

		Args:
			batch: A :obj:`list` of :obj:`dict`.
		"""
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        for k, v in zipped.items():
            collate_fn = cls.datapoint_attrs.get_collate_fn(k)
            if collate_fn and v:
                zipped[k] = collate_fn(batch=v)
        return zipped
    
    # endregion
    
# endregion


# region Multimodal Dataset

class MultimodalDataset(Dataset, ABC):
    """We design this class to be a multimodal, multi-task, and multi-label
    dataset. It is designed to be flexible and extensible to support various
    types of datasets.
    
    Attributes:
        datapoint_attrs: A :obj:`dict` of datapoint attributes with the keys
            are the attribute names and the values are the attribute types.
            Must contain: {``'image'``: :obj:`ImageAnnotation`}. Note that to
            comply with :obj:`albumentations.Compose`, we will treat the first
            key as the main image attribute.
    
    Args:
        depth_source: The source of the depth data. Default: ``None``.
    """
    
    def __init__(
        self,
        depth_source: Literal[*DEPTH_DATA_SOURCES] = "dav2_vitb_g",
        *args, **kwargs
    ):
        if depth_source not in DEPTH_DATA_SOURCES:
            raise ValueError(f"`depth_source` must be one of "
                             f"{DEPTH_DATA_SOURCES}, but got {depth_source}.")
        self.depth_source = depth_source
        
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index: int) -> dict:
        """Returns a dictionary containing the datapoint and metadata at the
        given :obj:`index`.
        """
        # Get datapoint at the given index
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        # Transform
        if self.transform:
            main_attr      = self.main_attribute
            args           = {k: v for k, v in datapoint.items() if v is not None}
            args["image"]  = args.pop(main_attr)
            transformed    = self.transform(**args)
            transformed[main_attr] = transformed.pop("image")
            datapoint     |= transformed
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = self.datapoint_attrs.get_tensor_fn(k)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(v, keepdim=False, normalize=True)
        # Return
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        """Return the total number of datapoints in the dataset."""
        return len(self.datapoints[self.main_attribute])
    
    def init_transform(self, transform: A.Compose | Any = None):
        super().init_transform(transform=transform)
        # Add additional targets
        if isinstance(self.transform, A.Compose):
            additional_targets = self.datapoint_attrs.albumentation_target_types()
            additional_targets.pop(self.main_attribute, None)
            additional_targets.pop("meta", None)
            self.transform.add_targets(additional_targets)
    
    def init_data(self, cache_data: bool = False):
        """Initialize data."""
        # Get image from disk or cache
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.get_data()
            self.get_multimodal_data()
            
        # Filter and verify data
        self.filter_data()
        self.verify_data()
        
        # Cache data
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            pathlib.delete_cache(cache_file)
            
    def get_multimodal_data(self):
        """Get multimodal data."""
        print(self.datapoint_attrs.values())
        print(DepthMapAnnotation)
        if "ref_image" in self.datapoint_attrs.keys():
            self.get_reference_image()
        if DepthMapAnnotation in self.datapoint_attrs.values() and self.depth_source is not None:
            self.get_depth_map()
    
    def get_reference_image(self):
        """Get reference image."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])
        
        if len(ref_images) == 0:
            ref_images: list[ImageAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref/")
                    ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
            self.datapoints["ref_image"] = ref_images
    
    def get_depth_map(self):
        """Get depth map."""
        images     = self.datapoints.get("image",     [])
        depths     = self.datapoints.get("depth",     [])
        ref_images = self.datapoints.get("ref_image", [])
        ref_depths = self.datapoints.get("ref_depth", [])
        
        # Depth images
        if len(images) > 0 and len(depths) == 0:
            depths: list[DepthMapAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} {self.split_str} depth maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/{root_name}_{self.depth_source}/")
                    depths.append(
                        DepthMapAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["depth"] = depths
            
        # Reference depth images
        if len(ref_images) > 0 and len(ref_depths) == 0:
            ref_depths: list[DepthMapAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = ref_images,
                    description = f"Listing {self.__class__.__name__} {self.split_str} reference depth maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/{root_name}_{self.depth_source}/")
                    ref_depths.append(
                        DepthMapAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["ref_depth"] = ref_depths
    
    def filter_data(self):
        """Filter unwanted datapoints."""
        pass
    
    def verify_data(self):
        """Verify dataset."""
        if self.__len__() <= 0:
            raise RuntimeError(f"No datapoints in the dataset.")
        for k, v in self.datapoints.items():
            if k not in self.datapoint_attrs:
                raise RuntimeError(f"Attribute ``{k}`` has not been defined in "
                                   f"`datapoint_attrs`. If this is not an error, "
                                   f"please define the attribute in the class.")
            if self.datapoint_attrs[k]:
                if v is None:
                    raise RuntimeError(f"No ``{k}`` attributes has been defined.")
                if v is not None and len(v) != self.__len__():
                    raise RuntimeError(f"Number of {k} attributes ({len(v)}) "
                                       f"does not match the number of "
                                       f"datapoints ({self.__len__()}).")
        if self.verbose:
            console.log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    def reset(self):
        """Reset and start over."""
        self.index = 0
    
    def close(self):
        """Stop and release."""
        pass
    
    def get_datapoint(self, index: int) -> dict:
        """Get a datapoint at the given :obj:`index`."""
        datapoint = self.new_datapoint
        for k, v in self.datapoints.items():
            if v is not None and v[index] and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
        return datapoint
    
    def get_meta(self, index: int) -> dict:
        """Get metadata at the given :obj:`index`. By default, we will use the
        first attribute in :obj:`datapoint_attrs` as the main image attribute.
        """
        return self.datapoints[self.main_attribute][index].meta
    
# endregion
