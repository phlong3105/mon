#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple dataset types used in vision tasks and
datasets. We try to support all possible data types: :class:`torch.Tensor`,
:class:`numpy.ndarray`.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`_
"""

from __future__ import annotations

__all__ = [
    "COCODetectionDataset", "DataModule", "ImageClassificationDataset",
    "ImageClassificationDirectoryTree", "ImageDetectionDataset",
    "ImageDirectoryDataset", "ImageEnhancementDataset", "ImageLabelsDataset",
    "ImageSegmentationDataset", "LabeledImageDataset", "LabeledVideoDataset",
    "UnlabeledImageDataset", "UnlabeledVideoDataset", "VOCDetectionDataset",
    "VideoClassificationDataset", "VideoDetectionDataset", "VideoLabelsDataset",
    "YOLODetectionDataset",
]

import uuid
from abc import ABC, abstractmethod
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch

from mon.coreml import data
from mon.foundation import console, file_handler, pathlib, rich
from mon.globals import BBoxFormat
from mon.vision import image as mi
from mon.vision.dataset.base import label

DataModule = data.DataModule


# region Unlabeled Dataset

class UnlabeledImageDataset(data.UnlabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    images. This is mainly used for unsupervised learning tasks.
    
    See Also: :class:`mon.coreml.data.dataset.UnlabeledDataset`.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ['train', 'val', 'test',
            'predict']. Defaults to 'train'.
        image_size: The desired image size in HW format. Defaults to 256.
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on both the input and target. We
            use `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`_
        to_tensor: If True, convert input and target to :class:`torch.Tensor`.
            Defaults to False.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster loading
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            to_tensor = to_tensor,
            verbose   = verbose,
            *args, **kwargs
        )
        self.image_size  = mi.get_hw(size=image_size)
        self.classlabels = data.ClassLabels.from_value(value=classlabels)
        self.images: list[label.ImageLabel] = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.get_images()
        else:
            cache = torch.load(cache_file)
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
    
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        dict | None
    ]:
        """Return the image and metadata, optionally transformed by the
        respective transforms.
        """
        image = self.images[index].data
        meta  = self.images[index].meta
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        if self.to_tensor:
            image = mi.to_tensor(image=image, keepdim=False, normalize=True)
            
        return image, None, meta
        
    @abstractmethod
    def get_images(self):
        """Get image files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    
    def verify(self):
        """Verify and check data."""
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        console.log(f"Number of samples: {len(self.images)}.")
    
    def cache_data(self, path: pathlib.Path):
        """Cache data to :param:`path`."""
        cache = {"images": self.images}
        torch.save(cache, str(path))
    
    def cache_images(self):
        """Cache images into memory for a faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
    
    def reset(self):
        """Reset and start over."""
        pass
    
    def close(self):
        """Stop and release."""
        pass
    
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: A list of tuples of (input, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        
        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
            input = torch.cat(input, dim=0)
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in input):
            input = np.array(input)
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in input):
            input = np.concatenate(input, axis=0)
        else:
            raise ValueError(
                f"input's number of dimensions must be between 2 and 4."
            )
        
        target = None
        return input, target, meta


class UnlabeledVideoDataset(data.UnlabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    videos. This is mainly used for unsupervised learning tasks.
    
    See Also: :class:`mon.coreml.data.dataset.UnlabeledDataset`.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ['train', 'val', 'test',
            'predict']. Defaults to 'train'.
        image_size: The desired image size in HW format. Defaults to 256.
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        max_samples: Only process a certain number of samples. Defaults to None.
        transform: Transformations performing on both the input and target. We
            use `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`_
        to_tensor: If True, convert input and target to :class:`torch.Tensor`.
            Defaults to False.
        api_preference: Preferred Capture API backends to use. Can be used to
            enforce a specific reader implementation. Two most used options are:
            [cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900]. See more:
            https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base
            .htmlggaeb8dd9c89c10a5c63c139bf7c4f5704da7b235a04f50a444bc2dc72f5ae394aaf
            Defaults to cv2.CAP_FFMPEG.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root          : pathlib.Path,
        split         : str                     = "train",
        image_size    : int | list[int]         = 256,
        classlabels   : data.ClassLabels | None = None,
        max_samples   : int              | None = None,
        transform     : A.Compose        | None = None,
        to_tensor     : bool                    = False,
        api_preference: int                     = cv2.CAP_FFMPEG,
        verbose       : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root        = root,
            split       = split,
            classlabels = classlabels,
            transform   = transform,
            to_tensor   = to_tensor,
            verbose     = verbose,
            *args, **kwargs
        )
        self.image_size     = mi.get_hw(size=image_size)
        self.api_preference = api_preference
        self.source         = pathlib.Path("")
        self.video_capture  = None
        self.index          = 0
        self.max_samples    = max_samples
        self.num_images     = 0
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
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        dict | None
    ]:
        """Return the image and metadata, optionally transformed by the
        respective transforms.
        """
        if index >= self.num_images:
            self.close()
            raise StopIteration
        else:
            if isinstance(self.video_capture, cv2.VideoCapture):
                ret_val, image = self.video_capture.read()
                rel_path       = self.source.name
            else:
                raise RuntimeError(f"video_capture has not been initialized.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            meta  = {"rel_path": rel_path}

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            if self.to_tensor:
                image = mi.to_tensor(image=image, keepdim=False, normalize=True)
            
            return image, None, meta
    
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
        if self.source.is_video_file():
            self.video_capture = cv2.VideoCapture(str(self.source), self.api_preference)
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
        console.log(f"Number of samples: {self.num_images}.")

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
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        
        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
            input = torch.cat(input, dim=0)
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in input):
            input = np.array(input)
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in input):
            input = np.concatenate(input, axis=0)
        else:
            raise ValueError(
                f"input's number of dimensions must be between 2 and 4."
            )
        
        target = None
        return input, target, meta


class ImageDirectoryDataset(UnlabeledImageDataset):
    """A directory of images starting from a root directory.
    
    See Also: :class:`UnlabeledImageDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )
        
    def get_images(self):
        """Get image files."""
        if not self.root.is_dir():
            raise ValueError(
                f"root must be a valid directory, but got {self.root}."
            )
        
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                pattern.rglob("*"),
                description=f"[bright_yellow]Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                if path.is_image_file():
                    self.images.append(label.ImageLabel(path=path))
                    
    def filter(self):
        """Filter unwanted samples."""
        pass

# endregion


# region Labeled Dataset

class LabeledImageDataset(data.LabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    images.
    
    See Also: :class:`mon.coreml.data.dataset.LabeledDataset`.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ['train', 'val', 'test',
            'predict']. Defaults to 'train'.
        image_size: The desired image size in HW format. Defaults to 256.
        classlabels: :class:`mon.coreml.ClassLabels` object. Defaults to None.
        transform: Transformations performing on both the input and target.
        to_tensor: If True, convert input and target to :class:`torch.Tensor`.
            Defaults to False.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            to_tensor = to_tensor,
            verbose   = verbose,
            *args, **kwargs
        )
        self.image_size  = mi.get_hw(size=image_size)
        self.classlabels = data.ClassLabels.from_value(value=classlabels)
        self.images: list[label.ImageLabel] = []
        if not hasattr(self, "labels"):
            self.labels = []
        
        cache_file = self.root / f"{self.split}.cache"
        if cache_data or not cache_file.is_file():
            self.get_images()
            self.get_labels()
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
        torch.Tensor | np.ndarray,
        Any,
        dict | None
    ]:
        """Return the image, ground-truth, and metadata, optionally transformed
        by the respective transforms.
        """
        pass
    
    @abstractmethod
    def get_images(self):
        """Get image files."""
        pass

    @abstractmethod
    def get_labels(self):
        """Get label files."""
        pass

    @abstractmethod
    def filter(self):
        """Filter unwanted samples."""
        pass

    def verify(self):
        """Verify and check data."""
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        if not len(self.images) == len(self.labels):
            raise RuntimeError(
                f"Number of images and labels must be the same, but got "
                f"{len(self.images)} and {len(self.labels)}."
            )
        console.log(f"Number of {self.split} samples: {len(self.images)}.")
        
    def cache_data(self, path: pathlib.Path):
        """Cache data to :param:`path`."""
        cache = {
            "images": self.images,
            "labels": self.labels,
        }
        torch.save(cache, str(path))
        console.log(f"Cache data to: {path}")
    
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
    

class LabeledVideoDataset(data.LabeledDataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    videos.
    
    See Also: :class:`mon.coreml.data.dataset.LabeledDataset`.
    """
    pass

# endregion


# region Classification

class ImageClassificationDataset(LabeledImageDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated classification labels stored in a simple JSON format.
    
    See Also: :class:`LabeledImageDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        self.labels: list[label.ClassificationLabel] = []
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )
    
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | list,
        dict | None
    ]:
        """Return the image, ground-truth, and metadata, optionally transformed
        by the respective transforms.
        """
        image = self.images[index].data
        label = self.labels[index].data
        meta  = self.images[index].meta

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        if self.to_tensor:
            image = mi.to_tensor(image=image, keepdim=False, normalize=True)
            label = torch.Tensor(label)
            
        return image, label, meta
        
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")

    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        
        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
            input = torch.cat(input, dim=0)
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in input):
            input = np.array(input)
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in input):
            input = np.concatenate(input, axis=0)
        else:
            raise ValueError(
                f"input's number of dimensions must be between 2 and 4."
            )
        
        if all(isinstance(t, torch.Tensor) for t in target):
            target = torch.cat(target, dim=0)
        elif all(isinstance(t, np.ndarray) for t in target):
            target = np.concatenate(target, axis=0)
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
    
    def get_images(self):
        """Get image files."""
        pass

    def get_labels(self):
        """Get label files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass

# endregion


# region Object Detection

class ImageDetectionDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images, and a
    set of associated detections.
    
    See Also: :class:`LabeledImageDataset`.
    
    Args:
        bbox_format: A bounding box format specified in :class:`BBoxFormat`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        bbox_format : BBoxFormat              = BBoxFormat.XYXY,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        self.bbox_format = BBoxFormat.from_value(value=bbox_format)
        if isinstance(transform, A.Compose):
            if "bboxes" not in transform.processors:
                transform = A.Compose(
                    transforms  = transform.transforms,
                    bbox_params = A.BboxParams(format=str(self.bbox_format.value))
                )
        self.labels: list[label.DetectionsLabel] = []
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )
        
    def __getitem__(self, index: int) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        dict | None
    ]:
        """Return the image, ground-truth, and metadata, optionally transformed
        by the respective transforms.
        """
        image  = self.images[index].data
        bboxes = self.labels[index].data
        meta   = self.images[index].meta
        
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes)
            image       = transformed["image"]
            bboxes      = transformed["bboxes"]
        if self.to_tensor:
            image  = mi.to_tensor(image=image, keepdim=False, normalize=True)
            bboxes = torch.Tensor(bboxes)
            
        return image, bboxes, meta
        
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
    
    def filter(self):
        """Filter unwanted samples."""
        pass
        
    @staticmethod
    def collate_fn(batch) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | list | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed
        
        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
            input = torch.cat(input, dim=0)
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in input):
            input = np.array(input)
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in input):
            input = np.concatenate(input, axis=0)
        else:
            raise ValueError(
                f"input's number of dimensions must be between 2 and 4."
            )
        
        for i, l in enumerate(target):
            l[:, -1] = i  # add target image index for build_targets()
        return input, target, meta

    
class VideoDetectionDataset(LabeledVideoDataset, ABC):
    """The base class for datasets that represent a collection of videos and a
    set of associated video detections.
    
    See Also: :class:`ImageDetectionDataset`.
    """
    pass


class COCODetectionDataset(ImageDetectionDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated object detections saved in `COCO Object Detection Format
    <https://cocodataset.org/#format-data>`.
    
    See Also: :class:`ImageDetectionDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        bbox_format : BBoxFormat              = BBoxFormat.XYXY,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            bbox_format  = bbox_format,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )
    
    def get_labels(self):
        """Get label files."""
        json_file = self.annotation_file()
        if not json_file.is_json_file():
            raise ValueError(
                f"json_file must be a valid path to a .json file, but got "
                f"{json_file}."
            )
        json_data = file_handler.read_from_file(json_file)
        if not isinstance(json_data, dict):
            raise TypeError(
                f"json_data must be a dict, but got {type(json_data)}."
            )
        
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
            self.images[index].coco_url      = img.get("coco_url"     , "")
            self.images[index].flickr_url    = img.get("flickr_url"   , "")
            self.images[index].license       = img.get("license"      , 0)
            self.images[index].date_captured = img.get("date_captured", "")
            self.images[index].shape         = (height, width, 3)
        
        for ann in annotations:
            id          = ann.get("id"         , uuid.uuid4().int)
            image_id    = ann.get("image_id"   , None)
            bbox        = ann.get("bbox"       , None)
            category_id = ann.get("category_id", -1)
            area        = ann.get("area"       , 0)
            iscrowd     = ann.get("iscrowd"    , False)
        
    @abstractmethod
    def annotation_file(self) -> pathlib.Path:
        """Return the path to json annotation file."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    

class VOCDetectionDataset(ImageDetectionDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated object detections saved in `PASCAL VOC format
    <https://host.robots.ox.ac.uk/pascal/VOC>`.
    
    See Also: :class:`ImageDetectionDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        bbox_format : BBoxFormat              = BBoxFormat.XYXY,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            bbox_format  = bbox_format,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )
    
    def get_labels(self):
        """Get label files."""
        files = self.annotation_files()
        
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        if not len(self.images) == len(files):
            raise RuntimeError(
                f"Number of images and files must be the same, but got "
                f"{len(self.images)} and {len(files)}."
            )
        
        self.labels: list[label.VOCDetectionsLabel] = []
        with rich.get_progress_bar() as pbar:
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
    def annotation_files(self) -> list[pathlib.Path]:
        """Return the path to json annotation files."""
        pass
    
    def filter(self):
        """Filter unwanted samples."""
        pass
    

class YOLODetectionDataset(ImageDetectionDataset, ABC):
    """The base class for labeled datasets consisting of images, and their
    associated object detections saved in `YOLO format`.
    
    See Also: :class:`ImageDetectionDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        bbox_format : BBoxFormat              = BBoxFormat.XYXY,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            bbox_format  = bbox_format,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )
    
    def get_labels(self):
        """Get label files."""
        files = self.annotation_files()
        
        if not len(self.images) > 0:
            raise RuntimeError(f"No images in dataset.")
        if not len(self.images) == len(files):
            raise RuntimeError(
                f"Number of images and files must be the same, but got "
                f"{len(self.images)} and {len(files)}."
            )
        
        self.labels: list[label.YOLODetectionsLabel] = []
        with rich.get_progress_bar() as pbar:
            for f in pbar.track(
                files,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                self.labels.append(label.YOLODetectionsLabel.from_file(path=f))
        
    @abstractmethod
    def annotation_files(self) -> list[pathlib.Path]:
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
    
    See Also: :class:`LabeledImageDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        self.labels: list[label.ImageLabel] = []
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        dict | None
    ]:
        """Return the image, ground-truth, and metadata, optionally transformed
        by the respective transforms.
        """
        image = self.images[index].data
        label = self.labels[index].data
        meta  = self.images[index].meta
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image       = transformed["image"]
            label       = transformed["mask"]
        if self.to_tensor:
            image = mi.to_tensor(image=image, keepdim=False, normalize=True)
            label = mi.to_tensor(image=label, keepdim=False, normalize=True)
            
        return image, label, meta
        
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.__name__} {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
    
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
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | list | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
            input = torch.cat(input, dim=0)
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in input):
            input = np.array(input)
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in input):
            input = np.concatenate(input, axis=0)
        else:
            raise ValueError(
                f"input's number of dimensions must be between 2 and 4."
            )
        
        if all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in target):
            target = torch.stack(target, dim=0)
        elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
            target = torch.cat(target, dim=0)
        elif all(isinstance(t, np.ndarray) and t.ndim == 3 for t in target):
            target = np.array(target)
        elif all(isinstance(t, np.ndarray) and t.ndim == 4 for t in target):
            target = np.concatenate(target, axis=0)
        else:
            raise ValueError(
                f"target's number of dimensions must be between 2 and 4."
            )
        
        return input, target, meta
    
# endregion


# region Segmentation

class ImageSegmentationDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images and a
    set of associated semantic segmentations.
    
    See Also: :class:`LabeledImageDataset`.
    """
    
    def __init__(
        self,
        root        : pathlib.Path,
        split       : str                     = "train",
        image_size  : int | list[int]         = 256,
        classlabels : data.ClassLabels | None = None,
        transform   : A.Compose        | None = None,
        to_tensor   : bool                    = False,
        cache_data  : bool                    = False,
        cache_images: bool                    = False,
        verbose     : bool                    = True,
        *args, **kwargs
    ):
        self.labels: list[label.SegmentationLabel] = []
        super().__init__(
            root         = root,
            split        = split,
            image_size   = image_size,
            classlabels  = classlabels,
            transform    = transform,
            to_tensor    = to_tensor,
            cache_data   = cache_data,
            cache_images = cache_images,
            verbose      = verbose,
            *args, **kwargs
        )

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | None,
        dict | None
    ]:
        """Return the image, ground-truth, and metadata, optionally transformed
        by the respective transforms.
        """
        image = self.images[index].data
        label = self.labels[index].data
        meta  = self.images[index].meta

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image       = transformed["image"]
            label       = transformed["mask"]
        if self.to_tensor:
            image = mi.to_tensor(image=image, keepdim=False, normalize=True)
            label = mi.to_tensor(image=label, keepdim=False, normalize=True)
            
        return image, label, meta
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.images)),
                description=f"Caching {self.__class__.__name__} {self.split} images"
            ):
                self.images[i].load(keep_in_memory=True)
        console.log(f"Images have been cached.")
        
        with rich.get_download_bar() as pbar:
            for i in pbar.track(
                range(len(self.labels)),
                description=f"Caching {self.__class__.__name__} {self.split} labels"
            ):
                self.labels[i].load(keep_in_memory=True)
        console.log(f"Labels have been cached.")
    
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
        torch.Tensor | np.ndarray,
        torch.Tensor | np.ndarray | list | None,
        list | None
    ]:
        """Collate function used to fused input items together when using
        :attr:`batch_size` > 1. This is used in the
        :class:`torch.utils.data.DataLoader` wrapper.
        
        Args:
            batch: a list of tuples of (input, target, meta).
        """
        input, target, meta = zip(*batch)  # Transposed

        if all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in input):
            input = torch.stack(input, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
            input = torch.cat(input, dim=0)
        elif all(isinstance(i, np.ndarray) and i.ndim == 3 for i in input):
            input = np.array(input)
        elif all(isinstance(i, np.ndarray) and i.ndim == 4 for i in input):
            input = np.concatenate(input, axis=0)
        else:
            raise ValueError(
                f"input's number of dimensions must be between 2 and 4."
            )
        
        if all(isinstance(t, torch.Tensor) and t.ndim == 3 for t in target):
            target = torch.stack(target, dim=0)
        elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
            target = torch.cat(target, dim=0)
        elif all(isinstance(t, np.ndarray) and t.ndim == 3 for t in target):
            target = np.array(target)
        elif all(isinstance(t, np.ndarray) and t.ndim == 4 for t in target):
            target = np.concatenate(target, axis=0)
        else:
            raise ValueError(
                f"target's number of dimensions must be between 2 and 4."
            )
        
        return input, target, meta

# endregion


# region Multitask

class ImageLabelsDataset(LabeledImageDataset, ABC):
    """The base class for datasets that represent a collection of images, and a
    set of associated multitask predictions.
    
    See Also: :class:`LabeledImageDataset`.
    """
    pass


class VideoLabelsDataset(LabeledVideoDataset, ABC):
    """The base class for datasets that represent a collection of videos, and a
    set of associated multitask predictions.
    
    See Also: :class:`LabeledImageDataset`.
    """
    pass

# endregion
