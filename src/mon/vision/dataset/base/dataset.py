#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple dataset types used in vision tasks and
datasets. We try to support all possible data types: :class:`torch.Tensor`,
:class:`np.ndarray`, or :class:`Sequence`, but we prioritize
:class:`torch.Tensor`.
"""

from __future__ import annotations

__all__ = [
    "COCODetectionDataset", "DataModule", "Dataset",
    "ImageClassificationDataset", "ImageClassificationDirectoryTree",
    "ImageDetectionDataset", "ImageDirectoryDataset", "ImageEnhancementDataset",
    "ImageLabelsDataset", "ImageSegmentationDataset", "LabeledDataset",
    "LabeledImageDataset", "LabeledVideoDataset", "UnlabeledDataset",
    "UnlabeledImageDataset", "UnlabeledVideoDataset", "VOCDetectionDataset",
    "VideoClassificationDataset", "VideoDetectionDataset", "VideoLabelsDataset",
    "YOLODetectionDataset",
]

import uuid
from abc import ABC, abstractmethod
from typing import Any

import cv2
import munch
import torch

from mon import core, coreimage as ci, coreml
from mon.vision import constant
from mon.vision.dataset.base import label
from mon.vision.transform import transform as t
from mon.vision.typing import (
    ClassLabelsType, DictType, Ints, PathsType, PathType, TransformsType,
    VisionBackendType,
)

Dataset          = coreml.Dataset
UnlabeledDataset = coreml.UnlabeledDataset
LabeledDataset   = coreml.LabeledDataset
DataModule       = coreml.DataModule


# region Helper Function
# TODO: Add default transform if None is given
def parse_transform(
    transform: TransformsType | None = None,
    shape    : Ints                  = (3, 256, 256),
) -> TransformsType:
    if transform is None \
        or (isinstance(transform, list | tuple) and len(transform) == 0):
        transform = [t.Resize(size=shape)]
    

# endregion


# region Unlabeled Dataset

class UnlabeledImageDataset(UnlabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    images. This is mainly used for unsupervised learning tasks.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name        : str,
        root        : PathType,
        split       : str                    = "train",
        shape       : Ints                   = (3, 256, 256),
        classlabels : ClassLabelsType | None = None,
        transform   : TransformsType  | None = None,
        transforms  : TransformsType  | None = None,
        cache_data  : bool                   = False,
        cache_images: bool                   = False,
        backend     : VisionBackendType      = constant.VISION_BACKEND,
        verbose     : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name       = name,
            root       = root,
            split      = split,
            shape      = shape,
            transform  = transform,
            transforms = transforms,
            verbose    = verbose,
            *args, **kwargs
        )
        self.backend     = constant.VisionBackend.from_value(backend)
        self.classlabels = coreml.ClassLabels.from_value(classlabels)
        self.images: list[label.ImageLabel] = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.list_images()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
        
        self.filter()
        self.verify()
        if cache_data or not cache_file.is_file():
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
    
    def __len__(self) -> int:
        """Return the length of :attr:`images`."""
        return len(self.images)
    
    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        DictType     | None
    ]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.

        Args:
            index: The index of the sample to be retrieved.

        Return:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Metadata of image.
        """
        input = self.images[index].tensor
        meta  = self.images[index].meta
        
        if self.transform is not None:
            input, *_ = self.transform(input=input,  target=None, dataset=self)
        if self.transforms is not None:
            input, *_ = self.transforms(input=input, target=None, dataset=self)
        return input, None, meta
        
    @abstractmethod
    def list_images(self):
        """List image files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    
    def verify(self):
        """Verify and check data."""
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        core.console.log(f"Number of samples: {len(self.images)}.")
    
    def cache_data(self, path: PathType):
        """Cache data to :param:`path`."""
        cache = {"images": self.images}
        torch.save(cache, str(path))
    
    def cache_images(self):
        """Cache images into memory for a faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        core.console.log(f"Images have been cached.")
    
    def reset(self):
        """Reset and start over."""
        pass
    
    def close(self):
        """Stop and release."""
        pass
    
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list         | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: A list of tuples of (input, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input, 0)
        else:
            raise ValueError(f"Expect 3 <= `input.ndim` <= 4.")

        if all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in target):
            target = torch.stack(target, 0)
        elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
            target = torch.cat(target, 0)
        else:
            target = None
        return input, target, meta


class UnlabeledVideoDataset(UnlabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    videos. This is mainly used for unsupervised learning tasks.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        max_samples: Only process a certain amount of samples. Defaults to None.
        transform: Transformations performing on the input.
        transforms: Transformations performing on both the input and target.
        api_preference: Preferred Capture API backends to use. Can be used to
            enforce a specific reader implementation. Two most used options are:
            [cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900]. See more:
            https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base.htmlggaeb8dd9c89c10a5c63c139bf7c4f5704da7b235a04f50a444bc2dc72f5ae394aaf
            Defaults to cv2.CAP_FFMPEG.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name          : str,
        root          : PathType,
        split         : str                    = "train",
        shape         : Ints                   = (3, 256, 256),
        classlabels   : ClassLabelsType | None = None,
        max_samples   : int             | None = None,
        transform     : TransformsType  | None = None,
        transforms    : TransformsType  | None = None,
        api_preference: int                    = cv2.CAP_FFMPEG,
        verbose       : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            root        = root,
            split       = split,
            shape       = shape,
            classlabels = classlabels,
            transform   = transform,
            transforms  = transforms,
            verbose     = verbose,
            *args, **kwargs
        )
        self.api_preference   = api_preference
        self.source: PathType = ""
        self.video_capture    = None
        self.index            = 0
        self.max_samples      = max_samples
        self.num_images       = 0
        self.list_source()
        self.init_video_capture()
        self.filter()
        self.verify()
    
    def __iter__(self):
        """Return an iterator object starting at index 0."""
        self.reset()
        return self
    
    def __len__(self) -> int:
        """An alias of :attr:`num_images`."""
        return self.num_images
    
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor, 
        torch.Tensor | None, 
        DictType     | None
    ]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.

        Args:
            index: The index of the sample to be retrieved.

        Return:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Metadata of image.
        """
        if index >= self.num_images:
            self.close()
            raise StopIteration
        else:
            if isinstance(self.video_capture, cv2.VideoCapture):
                ret_val, image = self.video_capture.read()
                rel_path       = self.source.name
            else:
                raise RuntimeError(
                    f":attr:`video_capture` has not been initialized."
                )

            image = image[:, :, ::-1]  # BGR to RGB
            image = ci.to_tensor(image=image, keepdim=False, normalize=True)
            input = image
            meta  = {"rel_path": rel_path}

            if self.transform is not None:
                input, *_ = self.transform(input=input,  target=None, dataset=self)
            if self.transforms is not None:
                input, *_ = self.transforms(input=input, target=None, dataset=self)
            return input, None, meta
    
    @property
    def num_images(self) -> int:
        return self._num_images
    
    @num_images.setter
    def num_images(self, num_images: int | None):
        self._num_images = num_images or 0
        
    @abstractmethod
    def list_source(self):
        """List the video file."""
        pass
    
    def init_video_capture(self):
        source = str(self.source)
        if core.is_video_file(path=source):
            self.video_capture = cv2.VideoCapture(source, self.api_preference)
            num_images         = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise IOError(r"Error when reading input stream or video file!")

        if self.num_images == 0:
            self.num_images = num_images
        if self.max_samples is not None and self.num_images > self.max_samples:
            self.num_images = self.max_samples
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    
    def verify(self):
        """Verify and check data."""
        if not self.num_images > 0:
            raise RuntimeError(f"No images in dataset.")
        core.console.log(f"Number of samples: {self.num_images}.")

    def reset(self):
        """Reset and start over."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """Stop and release the current :attr:`video_capture` object."""
        if isinstance(self.video_capture, cv2.VideoCapture) \
            and self.video_capture:
            self.video_capture.release()
            
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input, 0)
        else:
            raise ValueError(f"Expect 3 <= `input.ndim` <= 4.")
        
        if all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in target):
            target = torch.stack(target, 0)
        elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
            target = torch.cat(target, 0)
        else:
            target = None
        return input, target, meta


class ImageDirectoryDataset(UnlabeledImageDataset):
    """A directory of images starting from a root directory.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name        : str,
        root        : PathType,
        split       : str                    = "train",
        shape       : Ints                    = (3, 256, 256),
        classlabels : ClassLabelsType | None = None,
        transform   : TransformsType  | None = None,
        transforms  : TransformsType  | None = None,
        cache_data  : bool                   = False,
        cache_images: bool                   = False,
        backend     : VisionBackendType      = constant.VISION_BACKEND,
        verbose     : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name         = name,
            root         = root,
            split        = split,
            shape        = shape,
            classlabels  = classlabels,
            transform    = transform,
            transforms   = transforms,
            cache_data   = cache_data,
            cache_images = cache_images,
            backend      = backend,
            verbose      = verbose,
            *args, **kwargs
        )
        
    def list_images(self):
        """List image files."""
        assert isinstance(self.root, core.Path) and self.root.is_dir()
        
        with core.rich.progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                pattern.rglob("*"),
                description=f"[bright_yellow]Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                if path.is_image_file():
                    self.images.append(label.ImageLabel(path=path, backend=self.backend))
                    
    def filter(self):
        """Filter unwanted samples."""
        pass

# endregion


# region Labeled Dataset

class LabeledImageDataset(LabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            verbose          = verbose,
            *args, **kwargs
        )
        self.backend     = constant.VisionBackend.from_value(backend)
        self.classlabels = coreml.ClassLabels.from_value(classlabels)
        self.images: list[label.ImageLabel] = []
        if not hasattr(self, "labels"):
            self.labels = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.list_images()
            self.list_labels()
        else:
            cache       = torch.load(cache_file)
            self.images = cache["images"]
            self.labels = cache["labels"]
            
        self.filter()
        self.verify()
        if cache_data or not cache_file.is_file():
            self.cache_data(path=cache_file)
        if cache_images:
            self.cache_images()
    
    def __len__(self) -> int:
        """Return the length of :attr:`images`."""
        return len(self.images)
    
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        Any,
        DictType | None
    ]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index: The index of the sample to be retrieved.

        Return:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Target, depending on the label type.
            Metadata of image.
        """
        pass
    
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
        """Verify and check data."""
        if not (len(self.images) == len(self.labels) and len(self.images) > 0):
            raise RuntimeError(
                f"Number of images and labels must be the same. "
                f"But got: {len(self.images)} != {len(self.labels)}"
            )
        core.console.log(f"Number of {self.split} samples: {len(self.images)}.")
        
    def cache_data(self, path: PathType):
        """Cache data to :param:`path`."""
        cache = {
            "images": self.images,
            "labels": self.labels,
        }
        torch.save(cache, str(path))
        core.console.log(f"Cache data to: {path}")
    
    @abstractmethod
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        pass
    
    def reset(self):
        """Reset and start over."""
        pass
    
    def close(self):
        """Stop and release."""
        pass
    

class LabeledVideoDataset(LabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    videos.
    """
    pass

# endregion


# region Classification

class ImageClassificationDataset(LabeledImageDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated classification labels stored in a simple JSON format.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        shape: The desired datapoint shape preferably in a channel-last format.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.labels: list[label.ClassificationsLabel] = []
        if transform is None:
            transform = [
                t.Resize(size=shape),
            ]
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        int,
        DictType | None
    ]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index: The index of the sample to be retrieved.

        Return:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Classification labels.
            Metadata of image.
        """
        input  = self.images[index].tensor
        target = self.labels[index].tensor
        meta   = self.images[index].meta
        
        if self.transform is not None:
            input,  *_    = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_    = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        core.console.log(f"Images have been cached.")

    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input,  0)
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        
        if all(isinstance(t, torch.Tensor) for t in target):
            target = torch.cat(target, 0)
        else:
            target = None
        return input, target, meta
    

class VideoClassificationDataset(LabeledVideoDataset, ABC):
    """The base class for datasets that represent a collection of videos, and a
    set of associated classification labels.
    """
    pass


class ImageClassificationDirectoryTree(ImageClassificationDataset):
    """A directory tree whose sub-folders define an image classification
    dataset.
    """
    
    def list_images(self):
        """List image files."""
        pass

    def list_labels(self):
        """List label files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass

# endregion


# region Object Detection

class ImageDetectionDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images, and a
    set of associated detections.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.labels: list[label.DetectionsLabel] = []
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        DictType | None
    ]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index: The index of the sample to be retrieved.

        Return:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Bounding boxes of shape [N, 7].
            Metadata of image.
        """
        input  = self.images[index].tensor
        target = self.labels[index].tensor
        meta   = self.images[index].meta

        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        core.console.log(f"Images have been cached.")
    
    def filter(self):
        """Filter unwanted samples."""
        pass
        
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input  = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input  = torch.cat(input,  0)
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        for i, l in enumerate(target):
            l[:, 0] = i  # add target image index for build_targets()
        return input, target, meta

    
class VideoDetectionDataset(LabeledVideoDataset, ABC):
    """The base class for datasets that represent a collection of videos and a
    set of associated video detections.
    """
    pass


class COCODetectionDataset(ImageDetectionDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated object detections saved in `COCO Object Detection Format
    <https://cocodataset.org/#format-data>`.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """List label files."""
        json_file = self.annotation_file()
        assert json_file.is_json_file()
        json_data = core.load_from_file(json_file)
        assert isinstance(json_data, dict | munch.Munch)
        
        info	    = json_data.get("info", 	   None)
        licenses    = json_data.get("licenses",    None)
        categories  = json_data.get("categories",  None)
        images	    = json_data.get("images",	   None)
        annotations = json_data.get("annotations", None)

        for img in images:
            id       = img.get("id",        uuid.uuid4().int)
            filename = img.get("file_name", "")
            height   = img.get("height",     0)
            width    = img.get("width",      0)
            index    = -1
            for idx, im in enumerate(self.images):
                if im.name == filename:
                    index = idx
                    break
            self.images[index].id            = id
            self.images[index].coco_url      = img.get("coco_url",      "")
            self.images[index].flickr_url    = img.get("flickr_url",    "")
            self.images[index].license       = img.get("license",       0 )
            self.images[index].date_captured = img.get("date_captured", "")
            self.images[index].shape         = (3, height, width)
        
        for ann in annotations:
            id          = ann.get("id",           uuid.uuid4().int)
            image_id    = ann.get("image_id",     None)
            bbox        = ann.get("bbox",         None)
            category_id = ann.get("category_id", -1)
            area        = ann.get("area",         0)
            iscrowd     = ann.get("iscrowd",      False)
        
    @abstractmethod
    def annotation_file(self) -> PathType:
        """Return the path to json annotation file."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    

class VOCDetectionDataset(ImageDetectionDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated object detections saved in `PASCAL VOC format
    <https://host.robots.ox.ac.uk/pascal/VOC>`.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """List label files."""
        files = self.annotation_files()
        if not (len(files) == len(self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of files and labels must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        self.labels: list[label.VOCDetectionsLabel] = []
        with core.rich.progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                self.labels.append(
                    label.VOCDetectionsLabel.from_file(
                        path        = f,
                        classlabels = self.classlabels
                    )
                )
                
    @abstractmethod
    def annotation_files(self) -> PathsType:
        """Return the path to json annotation files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    

class YOLODetectionDataset(ImageDetectionDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated object detections saved in `YOLO format`.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
        shape: The desired datapoint shape preferably in a channel-last format.
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_labels(self):
        """List label files."""
        files = self.annotation_files()
        if not (len(files) == len(self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of `images` and `labels` must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        self.labels: list[label.YOLODetectionsLabel] = []
        with core.rich.progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                self.labels.append(label.YOLODetectionsLabel.from_file(path=f))
        
    @abstractmethod
    def annotation_files(self) -> PathsType:
        """Return the path to json annotation files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    
# endregion


# region Image Enhancement

class ImageEnhancementDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images, and a
    set of associated enhanced images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
        shape: The desired datapoint shape preferably in a channel-last format.
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.labels: list[label.ImageLabel] = []
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor, 
        torch.Tensor,
        DictType | None
    ]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
            index: The index of the sample to be retrieved.

        Return:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Target of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Metadata of image.
        """
        input  = self.images[index].tensor
        target = self.labels[index].tensor
        meta   = self.images[index].meta

        if self.transform is not None:
            input,  *_     = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_     = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input,  target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
        
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        core.console.log(f"Images have been cached.")
        
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.__name__} {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        core.console.log(f"Labels have been cached.")
    
    def filter(self):
        """Filter unwanted samples."""
        keep = []
        for i, (img, lab) in enumerate(zip(self.images, self.labels)):
            if img.path.is_image_file() and lab.path.is_image_file():
                keep.append(i)
        self.images = [img for i, img in enumerate(self.images) if i in keep]
        self.labels = [lab for i, lab in enumerate(self.labels) if i in keep]
        
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input):
            input  = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input  = torch.cat(input, 0)
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
            )

        if all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in target):
            target = torch.stack(target, 0)
        elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
            target = torch.cat(target, 0)
        else:
            target = None
        return input, target, meta
    
# endregion


# region Segmentation

class ImageSegmentationDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images and a
    set of associated semantic segmentations.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
        shape: The desired datapoint shape preferably in a channel-last format.
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.labels: list[label.SegmentationLabel] = []
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
            index: The index of the sample to be retrieved.
          
        Return:
            input: Input sample, optionally transformed by the respective
                transforms.
            target: Semantic segmentation mask, optionally transformed by the
                respective transforms.
            meta: Metadata of image.
        """
        input  = self.images[index].tensor
        target = self.labels[index].tensor
        meta   = self.images[index].meta

        if self.transform is not None:
            input,  *_ = self.transform(input=input, target=None, dataset=self)
        if self.target_transform is not None:
            target, *_ = self.target_transform(input=target, target=None, dataset=self)
        if self.transforms is not None:
            input, target = self.transforms(input=input, target=target, dataset=self)
        return input, target, meta
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        core.console.log(f"Images have been cached.")
        
        with core.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.__name__} {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        core.console.log(f"Labels have been cached.")
    
    def filter(self):
        """Filter unwanted samples."""
        keep = []
        for i, (img, lab) in enumerate(zip(self.images, self.labels)):
            if img.path.is_image_file() and lab.path.is_image_file():
                keep.append(i)
        self.images = [img for i, img in enumerate(self.images) if i in keep]
        self.labels = [lab for i, lab in enumerate(self.labels) if i in keep]
        
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        """
        input, target, meta = zip(*batch)  # Transposed
        if all(i.ndim == 3 for i in input):
            input  = torch.stack(input,  0)
        elif all(i.ndim == 4 for i in input):
            input  = torch.cat(input,  0)
        else:
            raise ValueError(
                f"Expect 3 <= `input.ndim` and `target.ndim` <= 4."
            )
        
        if all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in target):
            target = torch.stack(target, 0)
        elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
            target = torch.cat(target, 0)
        else:
            target = None
        return input, target, meta

# endregion


# region Multitask

class ImageLabelsDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images, and a
    set of associated multitask predictions.
    """
    pass


class VideoLabelsDataset(LabeledVideoDataset, ABC):
    """The base class for datasets that represent a collection of videos, and a
    set of associated multitask predictions.
    """
    pass

# endregion
