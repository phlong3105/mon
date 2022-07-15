#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""One dataset types.

Here is the taxonomy of the datasets:

    torch.utils.data.Dataset
      |
      |__ Dataset
            |
            |__ UnlabeledDataset
            |     |__ UnlabeledImageDataset
            |     |     |__ ImageDirectoryDataset
            |     |
            |     |__ UnlabeledVideoDataset
            |
            |__ LabeledDataset
                  |__ LabeledImageDataset
                  |     |__ ImageClassificationDataset
                  |     |     |__ ImageClassificationDirectoryDataset
                  |     |
                  |     |__ ImageDetectionDataset
                  |     |     |__ COCODetectionDataset
                  |     |     |__ KITTIDetectionDataset
                  |     |     |__ VOCDetectionDataset
                  |     |
                  |     |__ ImageEnhancementDataset
                  |     |__ ImageSegmentationDataset
                  |     |__ ImageLabelsDataset
                  |
                  |__ LabeledVideoDataset
                        |__ VideoClassificationDataset
                        |__ VideoDetectionDataset
                        |__ VideoLabelsDataset
"""

from __future__ import annotations

import inspect
import os.path
import sys
from abc import ABCMeta
from abc import abstractmethod
from glob import glob
from pathlib import Path
from typing import Any
from typing import Union

import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor

from one.core import Callable
from one.core import console
from one.core import download_bar
from one.core import Int3T
from one.core import is_image_file
from one.core import is_json_file
from one.core import load_file
from one.core import progress_bar
from one.core import VISION_BACKEND
from one.core import VisionBackend
from one.datasets.utils import BBox
from one.datasets.utils import ClassLabel
from one.datasets.utils import Image
from one.datasets.utils import VOCLabel


# MARK: - Base

class Dataset(data.Dataset, metaclass=ABCMeta):
    """Base class for making datasets. It is necessary to override the
    `__getitem__` and `__len__` method.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        transform       : Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
        transforms      : Union[Callable, None] = None,
        verbose         : bool                  = True,
        *args, **kwargs
    ):
        self.root             = root
        self.split            = split
        self.shape            = shape
        self.verbose          = verbose
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        """
        has_transforms         = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can be passed "
                "as argument."
            )

        self.transform        = transform
        self.target_transform = target_transform
        
        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        """
        
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        
        Args:
            index (int):
                Index.

        Returns:
            (Any):
                Sample and meta data, optionally transformed by the respective
                transforms.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    def _format_transform_repr(self, transform: Callable, head: str) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""


# MARK: - Unlabeled

class UnlabeledDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets that represent an unlabeled collection of data
    samples.
    """
    pass


class UnlabeledImageDataset(UnlabeledDataset, metaclass=ABCMeta):
    """Base class for datasets that represent an unlabeled collection of images.
    This is mainly used for unsupervised learning tasks.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """

    # MARK: Magic Function
    
    def __init__(
        self,
        root        : str,
        split       : str,
        shape       : Int3T,
        transform   : Union[Callable, None]     = None,
        transforms  : Union[Callable, None]     = None,
        cache_data  : bool                      = False,
        cache_images: bool                      = False,
        backend     : Union[VisionBackend, str] = VISION_BACKEND,
        verbose     : bool                      = True,
        *args, **kwargs
    ):
        super().__init__(
            root       = root,
            split      = split,
            shape      = shape,
            transform  = transform,
            transforms = transforms,
            verbose    = verbose,
            *args, **kwargs
        )
        if isinstance(backend, str):
            backend = VisionBackend.from_str(backend)
        self.backend = backend if backend in VisionBackend else VISION_BACKEND

        self.images: list[Image] = []
        
        cache_file = os.path.join(self.root, f"{self.split}.cache")
        if cache_data or not os.path.isfile(cache_file):
            self.list_images()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]

        self.filter()
        self.verify()
        if cache_data or not os.path.isfile(cache_file):
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
        
    def __getitem__(self, index: int) -> tuple[Tensor, dict]:
        """

		Args:
			index (int):
				Index.

		Returns:
			input (Tensor):
				Sample, optionally transformed by the respective transforms.
			meta (dict):
			    Meta data of image.
		"""
        item  = self.images[index]
        input = item.image if item.image is not None else item.load()
        meta  = item.meta
        
        if self.transform is not None:
            input = self.transform(input, self)
        if self.transforms is not None:
            input = self.transforms(input, None, self)
        return input, meta
        
    def __len__(self) -> int:
        return len(self.images)
        
    # MARK: Configure
    
    @abstractmethod
    def list_images(self):
        """List image files."""
        pass
    
    @abstractmethod
    def filter(self):
        """Filter unwanted samples."""
        pass
    
    def verify(self):
        """Verify and checking."""
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        console.log(f"Number of samples: {len(self.images)}.")
    
    def cache_data(self, path: Union[str, Path]):
        """Cache data to `path`."""
        cache = {"images": self.images}
        torch.save(cache, path)
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.images)),  description=f"[red]Caching {self.split} images"):
                gb += self.images[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of images have been cached" % (gb / 1E9))
            
    # MARK: Utils
    
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, list]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input, 0)
        else:
            raise ValueError(f"Each `input.ndim` must be 3 or 4.")
            
        return input, meta


class UnlabeledVideoDataset(UnlabeledDataset, metaclass=ABCMeta):
    """Base class for datasets that represent an unlabeled collection of video.
    This is mainly used for unsupervised learning tasks.
    """
    pass


class ImageDirectoryDataset(UnlabeledImageDataset):
    """A directory of images starting from `root` dir.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """

    # MARK: Magic Function
    
    def __init__(
        self,
        root        : str,
        split       : str,
        shape       : Int3T,
        transform   : Union[Callable, None]     = None,
        transforms  : Union[Callable, None]     = None,
        cache_data  : bool                      = False,
        cache_images: bool                      = False,
        backend     : Union[VisionBackend, str] = VISION_BACKEND,
        verbose     : bool                      = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            shape        = shape,
            transform    = transform,
            transforms   = transforms,
            cache_data   = cache_data,
            cache_images = cache_images,
            backend      = backend,
            verbose      = verbose,
            *args, **kwargs
        )
        
    # MARK: Configure
    
    def list_images(self):
        """List image files."""
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"`root` must be valid directory. But got: {self.root}."
            )
            
        with progress_bar() as pbar:
            for path in pbar.track(
                glob(os.path.join(self.root, self.split, "**/*"), recursive=True),
                description=f"[bright_yellow]Listing {self.split} images"
            ):
                if is_image_file(path):
                    self.images.append(Image(path=path, backend=self.backend))
                    
    def filter(self):
        """Filter unwanted samples."""
        pass


# MARK: - Labeled

class LabeledDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets that represent an unlabeled collection of data
    samples.
    """
    pass


class LabeledImageDataset(LabeledDataset, metaclass=ABCMeta):
    """Base class for datasets that represent an unlabeled collection of images.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_data      : bool                         = False,
        cache_images    : bool                         = False,
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            verbose          = verbose,
            *args, **kwargs
        )
        if isinstance(backend, str):
            backend = VisionBackend.from_str(backend)
        self.backend = backend if backend in VisionBackend else VISION_BACKEND
        
        if not isinstance(class_labels, (ClassLabel, str, Path)):
            class_labels = os.path.join(self.root, "class_labels.json")
        if is_json_file(class_labels):
            class_labels = ClassLabel.create_from_file(path=class_labels)
        self.class_labels = class_labels
        
        if hasattr(self, "images"):
            self.images: list[Image] = []
        if hasattr(self, "labels"):
            self.labels = []
        
        cache_file = os.path.join(self.root, f"{self.split}.cache")
        if cache_data or not os.path.isfile(cache_file):
            self.list_images()
            self.list_labels()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
            self.labels = cache["labels"]
            
        self.filter()
        self.verify()
        if cache_data or not os.path.isfile(cache_file):
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
    
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Any, Image]:
        """

		Args:
			index (int):
				Index.

		Returns:
			input (Tensor):
				Input sample, optionally transformed by the respective transforms.
			target (Any):
			    Target, depended on label type.
			meta (Image):
			    Meta data of image.
		"""
        pass
    
    def __len__(self) -> int:
        return len(self.images)
    
    # MARK: Configure
    
    @abstractmethod
    def list_images(self):
        """List image files."""
        pass

    @abstractmethod
    def list_labels(self):
        """List label files."""
        pass

    @abstractmethod
    def filter(self):
        """Filter unwanted samples."""
        pass

    def verify(self):
        """Verify and checking."""
        if not (len(self.images) == len(self.labels) and len(self.images) > 0):
            raise RuntimeError(
                f"Number of `images` and `labels` must be the same. "
                f"But got: {len(self.images)} != {len(self.labels)}"
            )
        console.log(f"Number of samples: {len(self.images)}.")
        
    def cache_data(self, path: Union[str, Path]):
        """Cache data to `path`."""
        cache = {
            "images": self.images,
            "labels": self.labels,
        }
        torch.save(cache, path)
    
    @abstractmethod
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        pass


class LabeledVideoDataset(LabeledDataset, metaclass=ABCMeta):
    """Base class for datasets that represent an unlabeled collection of video.
    """
    pass


# MARK: - Classification

class ImageClassificationDataset(LabeledImageDataset, metaclass=ABCMeta):
    """A labeled dataset consisting of images and their associated
    classification labels stored in a simple JSON format.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """

    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_data      : bool                         = False,
        cache_images    : bool                         = False,
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        self.labels: list[int] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def __getitem__(self, index: int) -> tuple[Tensor, int, dict]:
        """

		Args:
			index (int):
				Index.

		Returns:
			input (Tensor):
				Input sample, optionally transformed by the respective
				transforms.
			target (int):
				Classification labels.
			meta (dict):
				Meta data of image.
		"""
        item   = self.images[index]
        input  = item.image if item.image is not None else item.load()
        target = self.labels[index]
        meta   = item.meta
        
        if self.transform is not None:
            input  = self.transform(input, self)
        if self.target_transform is not None:
            target = self.target_transform(target, self)
        if self.transforms is not None:
            input, target = self.transforms(input, target, self)
        return input, target, meta
    
    # MARK: Configure
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.images)), description=f"[red]Caching {self.split} images"):
                gb += self.images[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of images have been cached" % (gb / 1E9))


class VideoClassificationDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """Base type for datasets that represent a collection of videos and a set
    of associated classification labels.
    """
    pass


class ImageClassificationDirectoryTree(ImageClassificationDataset):
    """A directory tree whose sub-folders define an image classification
    dataset.
    """
    
    # MARK: Configure
    
    def list_images(self):
        """List image files."""
        pass

    def list_labels(self):
        """List label files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
   

# MARK: - Detection

class ImageDetectionDataset(LabeledImageDataset, metaclass=ABCMeta):
    """Base class for datasets that represent a collection of images and a set
    of associated detections.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_images    : bool                         = False,
        cache_data      : bool                         = False,
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        self.labels: list[list[BBox]] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """

		Args:
			index (int):
				Index.

		Returns:
			input (Tensor):
				Input sample, optionally transformed by the respective transforms.
			target (Tensor):
			    Bounding boxes.
			meta (Image):
			    Meta data of image.
		"""
        item   = self.images[index]
        input  = item.image if item.image is not None else item.load()
        target = self.labels[index]
        target = np.array([b.label for b in target], dtype=np.float32)
        meta   = item.meta

        if self.transform is not None:
            input = self.transform(input, self)
        if self.target_transform is not None:
            target = self.target_transform(target, self)
        if self.transforms is not None:
            input, target = self.transforms(input, target, self)
        return input, target, meta
        
    # MARK: Configure
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.images)), description=f"[red]Caching {self.split} images"):
                gb += self.images[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of images have been cached" % (gb / 1E9))
           
    # MARK: Utils
    
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input):
            input  = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input  = torch.cat(input,  0)
        else:
            raise ValueError(f"Each `input.ndim` and `target.ndim` must be 3 or 4.")
        
        for i, l in enumerate(target):
            l[:, 0] = i  # add target image index for build_targets()
            
        return input, target, meta

    
class VideoDetectionDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """Base type for datasets that represent a collection of videos and a set
    of associated video detections.
    """
    pass


class COCODetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """A labeled dataset consisting of images and their associated object
    detections saved in `COCO Object Detection Format <https://cocodataset.org/#format-data>`.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_images    : bool                         = False,
        cache_data      : bool                         = False,
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    # MARK: Configure

    def list_labels(self):
        """List label files."""
        json = self.annotation_file()
        if not is_json_file(json):
            raise FileNotFoundError(
                f"`annotation_file` must be a `json` file. But got: {json}."
            )
        json_data = load_file(json)
    
    @abstractmethod
    def annotation_file(self) -> str:
        """Returns the path to `json` annotation file."""
        pass


class VOCDetectionDataset(ImageDetectionDataset, metaclass=ABCMeta):
    """A labeled dataset consisting of images and their associated object
    detections saved in `PASCAL VOC format <http://host.robots.ox.ac.uk/pascal/VOC>`.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_images    : bool                         = False,
        cache_data      : bool                         = False,
        
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    # MARK: Configure

    def list_labels(self):
        """List label files."""
        files = self.annotation_files()
        if not (len(files) == len(self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of `files` and `labels` must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        labels: list[VOCLabel] = []
        with progress_bar() as pbar:
            for f in pbar.track(files, description=f"[red]Listing {self.split} labels"):
                labels.append(VOCLabel.create_from_file(path=f, class_labels=self.class_labels))
        
        self.labels = labels
        
    @abstractmethod
    def annotation_files(self) -> list[str]:
        """Returns the path to `json` annotation files."""
        pass


# MARK: - Enhancement

class ImageEnhancementDataset(LabeledImageDataset, metaclass=ABCMeta):
    """Base type for datasets that represent a collection of images and a set
    of associated enhanced images.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_data      : bool                         = False,
        cache_images    : bool                         = False,
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """

		Args:
			index (int):
				Index.

		Returns:
			input (Tensor):
				Input sample, optionally transformed by the respective transforms.
			target (Tensor):
			    Enhance image.
			meta (Image):
			    Meta data of image.
		"""
        input  = self.images[index].image
        target = self.labels[index].image
        input  = self.images[index].load() if input  is None else input
        target = self.labels[index].load() if target is None else target
        meta   = self.images[index].meta
        
        if self.transform is not None:
            input = self.transform(input, self)
        if self.target_transform is not None:
            target = self.target_transform(target, self)
        if self.transforms is not None:
            input, target = self.transforms(input, target, self)
        return input, target, meta
        
    # MARK: Configure

    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.images)), description=f"[red]Caching {self.split} images"):
                gb += self.images[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of images have been cached" % (gb / 1E9))
        
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.labels)), description=f"[red]Caching {self.split} labels"):
                gb += self.labels[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of labels have been cached" % (gb / 1E9))
    
    # MARK: Utils
    
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(f"Each `input.ndim` and `target.ndim` must be 3 or 4.")
            
        return input, target, meta
    

# MARK: - Segmentation

class ImageSegmentationDataset(LabeledImageDataset, metaclass=ABCMeta):
    """Base class for datasets that represent a collection of images and a set
    of associated semantic segmentations.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """
    
    # MARK: Magic Function
    
    def __init__(
        self,
        root            : str,
        split           : str,
        shape           : Int3T,
        class_labels    : Union[ClassLabel, str, Path] = None,
        transform       : Union[Callable, None]        = None,
        target_transform: Union[Callable, None]        = None,
        transforms      : Union[Callable, None]        = None,
        cache_data      : bool                         = False,
        cache_images    : bool                         = False,
        backend         : Union[VisionBackend, str]    = VISION_BACKEND,
        verbose         : bool                         = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        """

		Args:
			index (int):
				Index.

		Returns:
			input (Tensor):
				Input sample, optionally transformed by the respective transforms.
			target (Tensor):
			    Semantic segmentation mask.
			meta (Image):
			    Meta data of image.
		"""
        input  = self.images[index].image
        target = self.labels[index].image
        input  = self.images[index].load() if input  is None else input
        target = self.labels[index].load() if target is None else target
        meta   = self.images[index].meta

        if self.transform is not None:
            input = self.transform(input, self)
        if self.target_transform is not None:
            target = self.target_transform(target, self)
        if self.transforms is not None:
            input, target = self.transforms(input, target, self)
        return input, target, meta
    
    # MARK: Configure

    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.images)), description=f"[red]Caching {self.split} images"):
                gb += self.images[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of images have been cached" % (gb / 1E9))
        
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            for i in pbar.track(range(len(self.labels)), description=f"[red]Caching {self.split} labels"):
                gb += self.labels[i].load(keep_in_memory=True).nbytes
        console.log(f"%.1fGB of labels have been cached" % (gb / 1E9))
    
    # MARK: Utils
    
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, list]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the `DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(f"Each `input.ndim` and `target.ndim` must be 3 or 4.")
            
        return input, target, meta


# MARK: - Multitask

class ImageLabelsDataset(LabeledImageDataset, metaclass=ABCMeta):
    """Base class for datasets that represent a collection of images and a set
    of associated multitask predictions.
    """
    pass


class VideoLabelsDataset(LabeledVideoDataset, metaclass=ABCMeta):
    """Base class for datasets that represent a collection of videos and a set
    of associated multitask predictions.
    """
    pass


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
