#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.dataset.base.dataset` module implements multiple dataset
types used in vision tasks and datasets. We try to support all possible data
types: :class:`torch.Tensor`, :class:`np.ndarray`, or :class:`Sequence`, but we
prioritize :class:`torch.Tensor`.
"""

from __future__ import annotations

__all__ = [
    "COCODetectionDataset", "ImageClassificationDataset",
    "ImageClassificationDirectoryTree", "ImageDetectionDataset",
    "ImageDirectoryDataset", "ImageEnhancementDataset", "ImageLabelsDataset",
    "ImageSegmentationDataset", "LabeledDataset", "LabeledImageDataset",
    "LabeledVideoDataset", "UnlabeledDataset", "UnlabeledImageDataset",
    "UnlabeledVideoDataset", "VOCDetectionDataset",
    "VideoClassificationDataset", "VideoDetectionDataset", "VideoLabelsDataset",
    "YOLODetectionDataset",
]

import uuid
from abc import ABC, abstractmethod
from typing import Any

import cv2
import munch
import torch

from mon import coreimage as ci, coreml, foundation
from mon.foundation import console
from mon.vision.dataset.base import label
from mon.vision.typing import (
    ClassLabelsType, DictType, Image, Ints, PathsType, PathType, TransformsType,
    VisionBackendType,
)


# region Unlabeled Dataset

class UnlabeledDataset(coreml.Dataset, ABC):
    """:class:`UnlabeledDataset` implements the base class for all datasets
    that represent an unlabeled collection of data samples.
    """
    pass


class UnlabeledImageDataset(UnlabeledDataset, ABC):
    """:class:`UnlabeledDataset` implements the base class for datasets that
    represent an unlabeled collection of images. This is mainly used for
    unsupervised learning tasks.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name        : str,
        root        : PathType,
        split       : str,
        shape       : Ints,
        transform   : TransformsType | None = None,
        transforms  : TransformsType | None = None,
        cache_data  : bool                  = False,
        cache_images: bool                  = False,
        backend     : VisionBackendType     = ci.VISION_BACKEND,
        verbose     : bool                  = True,
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
        self.backend = ci.VisionBackend.from_value(backend)
        self.images: list[Image] = []
        
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
        """Returns the length of the images list."""
        return len(self.images)
    
    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        DictType     | None
    ]:
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.

        Args:
            index: The index of the sample to be retrieved.

        Returns:
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
        """Lists image files."""
        pass
    
    def filter(self):
        """Filters unwanted samples."""
        pass
    
    def verify(self):
        """Verifies and checks data."""
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        console.log(f"Number of samples: {len(self.images)}.")
    
    def cache_data(self, path: PathType):
        """Caches data to :param:`path`."""
        cache = {"images": self.images}
        torch.save(cache, str(path))
    
    def cache_images(self):
        """Caches images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
    
    def reset(self):
        """Resets and starts over."""
        pass
    
    def close(self):
        """Stops and releases."""
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


class UnlabeledVideoDataset(UnlabeledDataset, ABC):
    """:class:`UnlabeledVideoDataset` implements the base class for datasets
    that represent an unlabeled collection of video. This is mainly used for
    unsupervised learning tasks.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        num_images: Only process certain amount of samples. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
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
        split         : str,
        shape         : Ints,
        max_samples   : int            | None = None,
        transform     : TransformsType | None = None,
        transforms    : TransformsType | None = None,
        api_preference: int                   = cv2.CAP_FFMPEG,
        verbose       : bool                  = True,
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
        """Returns an iterator object starting at index 0."""
        self.reset()
        return self
    
    def __len__(self) -> int:
        """Returns the length of the images list."""
        return self.num_images
    
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor, 
        torch.Tensor | None, 
        DictType     | None
    ]:
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.

        Args:
            index: The index of the sample to be retrieved.

        Returns:
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
        """Lists video file."""
        pass
    
    def init_video_capture(self):
        source = str(self.source)
        if foundation.is_video_file(path=source):
            self.video_capture = cv2.VideoCapture(source, self.api_preference)
            num_images         = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise IOError(r"Error when reading input stream or video file!")

        if self.num_images == 0:
            self.num_images = num_images
        if self.max_samples is not None and self.num_images > self.max_samples:
            self.num_images = self.max_samples
    
    def filter(self):
        """Filters unwanted samples."""
        pass
    
    def verify(self):
        """Verifies and checks data."""
        if not self.num_images > 0:
            raise RuntimeError(f"No images in dataset.")
        console.log(f"Number of samples: {self.num_images}.")

    def reset(self):
        """Resets and starts over."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """Stops and releases the current :attr:`video_capture` object."""
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
    """:class:`ImageDirectoryDataset` implements a directory of images starting
    from a root directory.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name        : str,
        root        : PathType,
        split       : str,
        shape       : Ints,
        transform   : TransformsType | None = None,
        transforms  : TransformsType | None = None,
        cache_data  : bool                  = False,
        cache_images: bool                  = False,
        backend     : VisionBackendType     = ci.VISION_BACKEND,
        verbose     : bool                  = True,
        *args, **kwargs
    ):
        super().__init__(
            name         = name,
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
        
    def list_images(self):
        """Lists image files."""
        assert isinstance(self.root, foundation.Path) and self.root.is_dir()
        
        with foundation.rich.progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                pattern.rglob("*"),
                description=f"[bright_yellow]Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                if path.is_image_file():
                    self.images.append(Image(path=path, backend=self.backend))
                    
    def filter(self):
        """Filters unwanted samples."""
        pass

# endregion


# region Labeled Dataset

class LabeledDataset(coreml.Dataset, ABC):
    """:class:`LabeledDataset` implements the base class for datasets that
    represent an unlabeled collection of data samples.
    """
    pass


class LabeledImageDataset(LabeledDataset, ABC):
    """:class:`LabeledImageDataset` implements the base class for datasets that
    represent an unlabeled collection of images.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
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
        self.backend     = ci.VisionBackend.from_value(backend)
        self.classlabels = coreml.ClassLabels.from_value(classlabels)
        self.images: list[Image] = []
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
        """Returns the length of the images list."""
        return len(self.images)
    
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        Any,
        DictType | None
    ]:
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index: The index of the sample to be retrieved.

        Returns:
            Sample of shape [1, C, H, W], optionally transformed by the
                respective transforms.
            Target, depending on label type.
            Metadata of image.
        """
        pass
    
    @abstractmethod
    def list_images(self):
        """Lists image files."""
        pass

    @abstractmethod
    def list_labels(self):
        """Lists label files."""
        pass

    @abstractmethod
    def filter(self):
        """Filters unwanted samples."""
        pass

    def verify(self):
        """Verifies and checks data."""
        if not (len(self.images) == len(self.labels) and len(self.images) > 0):
            raise RuntimeError(
                f"Number of images and labels must be the same. "
                f"But got: {len(self.images)} != {len(self.labels)}"
            )
        console.log(f"Number of {self.split} samples: {len(self.images)}.")
        
    def cache_data(self, path: PathType):
        """Caches data to :param:`path`."""
        cache = {
            "images": self.images,
            "labels": self.labels,
        }
        torch.save(cache, str(path))
        console.log(f"Caches data to: {path}")
    
    @abstractmethod
    def cache_images(self):
        """Caches images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        pass
    
    def reset(self):
        """Resets and starts over."""
        pass
    
    def close(self):
        """Stops and releases."""
        pass
    

class LabeledVideoDataset(LabeledDataset, ABC):
    """:class:`LabeledVideoDataset` implements the base class for datasets that
    represent an unlabeled collection of video.
    """
    pass

# endregion


# region Classification

class ImageClassificationDataset(LabeledImageDataset, ABC):
    """:class:`LabeledVideoDataset` implements the base class for labeled
    datasets consisting of images and their associated classification labels
    stored in a simple JSON format.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.labels: list[label.ClassificationsLabel] = []
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
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index: The index of the sample to be retrieved.

        Returns:
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
        """Caches images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")

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
    """:class:`VideoClassificationDataset` implements the base class for
    datasets that represent a collection of videos and a set of associated
    classification labels.
    """
    pass


class ImageClassificationDirectoryTree(ImageClassificationDataset):
    """:class:`ImageClassificationDirectoryTree` implements a directory tree
    whose sub-folders define an image classification dataset.
    """
    
    def list_images(self):
        """Lists image files."""
        pass

    def list_labels(self):
        """Lists label files."""
        pass
    
    def filter(self):
        """Filters unwanted samples."""
        pass

# endregion


# region Object Detection

class ImageDetectionDataset(LabeledImageDataset, ABC):
    """:class:`ImageDetectionDataset` implements the base class for datasets
    that represent a collection of images and a set of associated detections.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
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
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
          index: The index of the sample to be retrieved.

        Returns:
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
        """Caches images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
    
    def filter(self):
        """Filters unwanted samples."""
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
    """:class:`VideoDetectionDataset` implements the base class for datasets
    that represent a collection of videos and a set of associated video
    detections.
    """
    pass


class COCODetectionDataset(ImageDetectionDataset, ABC):
    """:class:`COCODetectionDataset` implements the base class for labeled
    datasets consisting of images and their associated object detections saved
    in `COCO Object Detection Format <https://cocodataset.org/#format-data>`.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
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
        """Lists label files."""
        json_file = self.annotation_file()
        assert json_file.is_json_file()
        json_data = foundation.load_from_file(json_file)
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
        """Returns the path to json annotation file."""
        pass
    
    def filter(self):
        """Filters unwanted samples."""
        pass
    

class VOCDetectionDataset(ImageDetectionDataset, ABC):
    """:class:`VOCDetectionDataset` implements the base class for labeled
    datasets consisting of images and their associated object detections saved
    in `PASCAL VOC format <https://host.robots.ox.ac.uk/pascal/VOC>`.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
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
        """Lists label files."""
        files = self.annotation_files()
        if not (len(files) == len(self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of files and labels must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        self.labels: list[label.VOCDetectionsLabel] = []
        with foundation.rich.progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.classname} {self.split} labels"
            ):
                self.labels.append(
                    label.VOCDetectionsLabel.from_file(
                        path        = f,
                        classlabels = self.classlabels
                    )
                )
                
    @abstractmethod
    def annotation_files(self) -> PathsType:
        """Returns the path to json annotation files."""
        pass
    
    def filter(self):
        """Filters unwanted samples."""
        pass
    

class YOLODetectionDataset(ImageDetectionDataset, ABC):
    """:class:`YOLODetectionDataset` implements the base class for labeled
    datasets consisting of images and their associated object detections saved
    in `YOLO format`.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_images    : bool                   = False,
        cache_data      : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
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
        """Lists label files."""
        files = self.annotation_files()
        if not (len(files) == len(self.images) and len(self.images) > 0):
            raise RuntimeError(
                f"`Number of `images` and `labels` must be the same. "
                f"But got: {len(files)} != {len(self.labels)}"
            )
        
        self.labels: list[label.YOLODetectionsLabel] = []
        with foundation.rich.progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.classname} {self.split} labels"
            ):
                self.labels.append(label.YOLODetectionsLabel.from_file(path=f))
        
    @abstractmethod
    def annotation_files(self) -> PathsType:
        """Returns the path to json annotation files."""
        pass
    
    def filter(self):
        """Filters unwanted samples."""
        pass
    
# endregion


# region Image Enhancement

class ImageEnhancementDataset(LabeledImageDataset, ABC):
    """:class:`ImageEnhancementDataset` implements the base class for datasets
    that represent a collection of images and a set of associated enhanced
    images.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.labels: list[Image] = []
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
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
            index: The index of the sample to be retrieved.

        Returns:
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
        """Caches images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.classname} {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
    
    def filter(self):
        """Filters unwanted samples."""
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
        """ Collate function used to fused input items together when using
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
    """:class:`ImageSegmentationDataset` implement the base class for datasets
    that represent a collection of images and a set of associated semantic
    segmentations.
    
    Args:
        name: Dataset name.
        root: Root directory of dataset.
        split: Split to use. One of: ["train", "val", "test"].
        shape: Image of shape [H, W, C], [H, W], or [S, S].
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Functions/transforms that takes in an input sample and
            returns a transformed version. E.g, `transforms.RandomCrop`.
        target_transform: Functions/transforms that takes in a target and
            returns a transformed version.
        transforms: Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: Image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformsType  | None = None,
        target_transform: TransformsType  | None = None,
        transforms      : TransformsType  | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = ci.VISION_BACKEND,
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
        """Returns the sample and metadata, optionally transformed by the
        respective transforms.
        
        Args:
            index: The index of the sample to be retrieved.
          
        Returns:
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
        """Caches images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.classname} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with foundation.rich.download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.classname} {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
    
    def filter(self):
        """Filters unwanted samples."""
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
    """:class:`ImageLabelsDataset` implements the base class for datasets that
    represent a collection of images and a set of associated multitask
    predictions.
    """
    pass


class VideoLabelsDataset(LabeledVideoDataset, ABC):
    """:class:`VideoLabelsDataset` implements the base class for datasets that
    represent a collection of videos and a set of associated multitask
    predictions.
    """
    pass

# endregion
