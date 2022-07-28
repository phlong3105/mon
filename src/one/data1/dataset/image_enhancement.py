#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Template for (Supervised) Image Enhancement datasets.
"""

from __future__ import annotations

import os
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import torch
from joblib import delayed
from joblib import Parallel
from one.imgproc import resize
from one.io import create_dirs
from one.io import get_hash
from one.io import read_image
from sortedcontainers import SortedDict
from torch import Tensor
from torchvision.datasets import VisionDataset

from one.core import Augment_
from one.core import AUGMENTS
from one.core import console
from one.core import download_bar
from one.core import get_image_hw
from one.core import Int3T
from one.core import progress_bar
from one.core import to_tensor
from one.core import VISION_BACKEND
from one.core import VisionBackend
from one.data1.augment import BaseAugment
from one.data1.data_class import ClassLabels
from one.data1.data_class import ImageInfo
from one.data1.data_class import VisionData
from one.data1.label_handler import VisionDataHandler

__all__ = [
    "ImageEnhancementDataset",
]


# MARK: - ImageEnhancementDataset

class ImageEnhancementDataset(VisionDataset, metaclass=ABCMeta):
    """A base class for all (supervised) image enhancement datasets.

    Attributes:
        root (str):
            Dataset root directory that contains: train/val/test/...
            subdirectories.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        image_paths (list):
            List of all image files.
        eimage_paths (list):
            List of all target image files.
        label_paths (list):
            List of all label files that can provide extra info about the
            image and target image.
        custom_label_paths (list):
            List of all custom label files.
        data (list):
            List of all `VisionData` objects.
        class_labels (ClassLabels, optional):
            `ClassLabels` object contains all class-labels defined in the
            dataset. Default: `None`.
        shape (Dim3):
            Image shape as [H, W, C].
        caching_labels (bool):
            Should overwrite the existing cached labels?
        caching_images (bool):
            Cache images into memory for faster training.
        write_labels (bool):
            After loading images and labels for the first time, we will convert
            it to our custom data format and write to files.
        fast_dev_run (bool):
            Take a small subset of the data for fast debug (i.e, like unit
            testing).
        load_augment (dict):
            Augmented loading policy.
        augment (Augment_):
            Augmentation policy.
        transforms (callable, optional):
            Function/transform that takes input sample and its target as
            entry and returns a transformed version.
        transform (callable, optional):
            Function/transform that takes input sample as entry and returns
            a transformed version.
        target_transform (callable, optional):
            Function/transform that takes in the target and transforms it.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str,
        class_labels    : Optional[ClassLabels]   = None,
        shape           : Int3T                   = (720, 1280, 3),
        caching_labels  : bool                    = False,
        caching_images  : bool                    = False,
        write_labels    : bool                    = False,
        fast_dev_run    : bool                    = False,
        load_augment    : Optional[dict]          = None,
        augment         : Optional[Augment_]      = None,
        vision_backend  : Optional[VisionBackend] = None,
        transforms      : Optional[Callable]      = None,
        transform       : Optional[Callable]      = None,
        target_transform: Optional[Callable]      = None,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            transforms       = transforms,
            transform        = transform,
            target_transform = target_transform
        )
        self.split              = split
        self.class_labels       = class_labels
        self.shape              = shape
        self.caching_labels     = caching_labels
        self.caching_images     = caching_images
        self.write_labels       = write_labels
        self.fast_dev_run       = fast_dev_run
        self.load_augment       = load_augment
        self.augment            = augment
        
        if vision_backend in VisionBackend:
            self.vision_backend = vision_backend
        else:
            self.vision_backend = VISION_BACKEND
        
        self.image_paths        = []
        self.eimage_paths       = []
        self.label_paths        = []
        self.custom_label_paths = []
        self.data               = []
        
        # NOTE: List files
        self.list_files()
        # NOTE: Load class_labels
        self.load_class_labels()
        # NOTE: Load (and cache) data
        self.load_data()
        # NOTE: Post-load data
        self.post_load_data()
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Any:
        """Return a tuple of data item from the dataset."""
        items  = self.get_item(index=index)
        input  = items[0]
        target = items[1]
        rest   = items[2:]
        
        if self.augment is not None:
            input, target = self.augment(input=input, target=target)
        else:
            input  = to_tensor(input,  normalize=True)
            target = to_tensor(target, normalize=True) if (target is not None) else None
            
        if self.transform is not None:
            input  = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target) if (target is not None) else None
        if self.transforms is not None:
            input  = self.transforms(input)
            target = self.transforms(target) if (target is not None) else None
        return input, target, rest
        
    # MARK: Properties
    
    @property
    def augment(self) -> Optional[BaseAugment]:
        return self._augment
    
    @augment.setter
    def augment(self, augment: Optional[Augment_]):
        """Assign augment configs."""
        if isinstance(augment, BaseAugment):
            self._augment = augment
        elif isinstance(augment, dict):
            self._augment = AUGMENTS.build_from_dict(cfg=augment)
        else:
            self._augment = None

    @property
    def has_custom_labels(self) -> bool:
        """Check if we have custom label files. If `True`, then those files
        will be loaded. Else, load the raw data	from the dataset, convert
        them to our custom data format, and write to files.
        """
        return self._has_custom_labels

    @property
    def custom_label_paths(self) -> list:
        return self._custom_label_paths

    @custom_label_paths.setter
    def custom_label_paths(self, paths: Optional[list[int]]):
        self._custom_label_paths = paths
    
        n = len(self.custom_label_paths)
        self._has_custom_labels = (
            n > 0 and
            n == len(self.image_paths) and
            all(os.path.isfile(p) for p in self.custom_label_paths)
        )
    
    # MARK: List Files
    
    @abstractmethod
    def list_files(self):
        """List image and label files.

        Todos:
            - Look for image and label files in `split` directory.
            - We should look for our custom label files first.
            - If none is found, proceed to listing the images and raw labels'
              files.
            - After this method, these following attributes MUST be defined:
              `image_paths`, `target_paths`, `label_paths`, `custom_label_paths`.
        """
        pass
    
    # MARK: Load ClassLabels
    
    @abstractmethod
    def load_class_labels(self):
        """Load ClassLabels."""
        pass

    # MARK: Load Data
    
    def load_data(self):
        """Load and cache images, enhance images, and labels."""
        # NOTE: Cache labels
        """
        path              = self.eimage_paths[0]
        split_prefix      = path[: path.find(self.split)]
        cached_label_path = f"{split_prefix}{self.split}.cache"
        """
        cached_label_path = os.path.join(self.root, f"{self.split}.cache")
        if os.path.isfile(cached_label_path):
            cache = torch.load(cached_label_path)  # Load
            hash  = get_hash(
                self.image_paths + self.eimage_paths + self.label_paths
                + self.custom_label_paths
            )
            if self.caching_labels or cache["hash"] != hash:
                # Re-cache
                cache = self.cache_labels_multithreaded(path=cached_label_path)
        else:
            # Cache
            cache = self.cache_labels_multithreaded(path=cached_label_path)
        
        # NOTE: Get labels
        self.data = [cache[x] for x in self.image_paths]
        
        # NOTE: Cache images
        if self.caching_images:
            self.cache_images()
            self.cache_enhance_images()

    def cache_labels(self, path: str) -> dict:
        """Cache labels, check images and read shapes.

        Args:
            path (str):
                Path to save the cached labels.

        Returns:
            cache_labels (dict):
                Dictionary contains the labels (numpy array) and the
                original image shapes that were cached.
        """
        # NOTE: Load all labels in label files
        cache_labels = {}
        has_label    = len(self.label_paths) == len(self.image_paths)
        
        with progress_bar() as pbar:
            for i in pbar.track(
                range(len(self.image_paths)),
                description=f"[bright_yellow]Caching {self.split} labels"
            ):
                label_path = (
                    self.label_paths[i]
                    if (has_label and os.path.isfile(self.label_paths[i]))
                    else None
                )
                custom_label_path = (
                    self.custom_label_paths[i]
                    if (self.has_custom_labels
                        and os.path.isfile(self.custom_label_paths[i]))
                    else None
                )
                label = self.load_label(
                    image_path        = self.image_paths[i],
                    enhance_path      = self.eimage_paths[i],
                    label_path        = label_path,
                    custom_label_path = custom_label_path
                )
                cache_labels[self.image_paths[i]] = label
            
        # NOTE: Check for any changes btw the cached labels
        for (k, v) in cache_labels.items():
            if v.image_info.path != k:
                self.caching_labels = True
                break
        
        # NOTE: Write cache
        console.log(f"Labels has been cached to: {path}.")
        cache_labels["hash"] = get_hash(
            self.image_paths + self.eimage_paths + self.label_paths
            + self.custom_label_paths
        )
        torch.save(cache_labels, path)  # Save for next time
        return cache_labels
    
    def cache_labels_multithreaded(self, path: str) -> dict:
        """Cache labels, check images and read shapes with multi-threading.

        Args:
            path (str):
                Path to save the cached labels.

        Returns:
            cache_labels (dict):
                Dictionary contains the labels (numpy array) and the
                original image shapes that were cached.
        """
        # NOTE: Load all labels in label files
        cache_labels = {}
        total        = len(self.image_paths)
        has_label    = len(self.label_paths) == total
        
        with progress_bar() as pbar:
            task = pbar.add_task(
                f"[bright_yellow]Caching {self.split} labels", total=total
            )
            
            def cache_label(i):
                label_path = (
                    self.label_paths[i]
                    if (has_label and os.path.isfile(self.label_paths[i]))
                    else None
                )
                custom_label_path = (
                    self.custom_label_paths[i]
                    if (self.has_custom_labels
                        and os.path.isfile(self.custom_label_paths[i]))
                    else None
                )
                label = self.load_label(
                    image_path        = self.image_paths[i],
                    enhance_path      = self.eimage_paths[i],
                    label_path        = label_path,
                    custom_label_path = custom_label_path
                )
                cache_labels[self.image_paths[i]] = label
                pbar.update(task, advance=1)
            
            Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
                delayed(cache_label)(i) for i in range(total)
            )
        
        # NOTE: Check for any changes btw the cached labels
        cache_labels = SortedDict(cache_labels)
        for (k, v) in cache_labels.items():
            if v.image_info.path != k:
                self.caching_labels = True
                break
        
        # NOTE: Write cache
        console.log(f"Labels has been cached to: {path}.")
        cache_labels["hash"] = get_hash(
            self.image_paths + self.eimage_paths + self.label_paths
            + self.custom_label_paths
        )
        torch.save(cache_labels, path)  # Save for next time
        return cache_labels
        
    @abstractmethod
    def load_label(
        self,
        image_path       : str,
        enhance_path     : str,
        label_path       : Optional[str] = None,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            enhance_path (str):
                Enhance image file.
            label_path (str, optional):
                Label file. Default: `None`.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.
                
        Returns:
            data (VisionData):
                `VisionData` object.
        """
        pass
    
    def cache_images(self):
        """Cache images into memory for faster training (WARNING: large
        datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            # Should be max 10k images
            for i in pbar.track(
                range(len(self.image_paths)),
                description=f"[red]Caching {self.split} images"
            ):
                # image, hw_original, hw_resized
                (self.data[i].image,
                 self.data[i].image_info) = self.load_image(index=i)
                gb += self.data[i].image.nbytes
                # pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)
    
    def load_image(self, index: int) -> tuple[np.ndarray, ImageInfo]:
        """Load 1 image from dataset and preprocess image.

        Args:
            index (int):
                Image index.

        Returns:
            image (np.ndarray):
                Image.
            info (ImageInfo):
                `ImageInfo` object.
        """
        image = self.data[index].image
        info  = self.data[index].image_info
        
        if image is None:  # Not cached
            path  = self.image_paths[index]
            image = read_image(path, backend=self.vision_backend)  # RGB
            if image is None:
                raise ValueError(f"No image found at: {path}.")
            
            # NOTE: Resize image while keeping the image ratio
            h0, w0 = get_image_hw(image)
            image  = resize(image, self.shape)
            h1, w1 = get_image_hw(image)
            
            # NOTE: Assign image info if it has not been defined
            # (just to be sure)
            info        = ImageInfo.from_file(image_path=path, info=info)
            info.height = h1 if info.height != h1 else info.height
            info.width  = w1 if info.width  != w1 else info.width
            info.depth  = (image.shape[2] if info.depth != image.shape[2]
                           else info.depth)
        
        return image, info
    
    def cache_enhance_images(self):
        """Cache target images into memory for faster training (WARNING:
        large datasets may exceed system RAM).
        """
        gb = 0  # Gigabytes of cached images
        with download_bar() as pbar:
            # Should be max 10k images
            for i in pbar.track(
                range(len(self.eimage_paths)),
                description=f"Caching {self.split} enhance images"
            ):
                # image, hw_original, hw_resized
                (self.data[i].eimage,
                 self.data[i].eimage_info) = self.load_enhance_image(index=i)
                gb += self.data[i].eimage.nbytes
                # pbar.desc = "Caching target images (%.1fGB)" % (gb / 1E9)
    
    def load_enhance_image(self, index: int) -> tuple[np.ndarray, ImageInfo]:
        """Load 1 target image from dataset and preprocess image.

        Args:
            index (int):
                Enhance image index.

        Returns:
            image (np.ndarray):
                Target image.
            info (ImageInfo):
                `ImageInfo` object.
        """
        image = self.data[index].eimage
        info  = self.data[index].eimage_info
        
        if image is None:  # Not cached
            path  = self.eimage_paths[index]
            image = read_image(path, backend=self.vision_backend)  # RGB
            if image is None:
                raise ValueError(f"No enhanced image found at: {path}.")
            
            # NOTE: Resize image while keeping the image ratio
            h0, w0 = get_image_hw(image)
            image  = resize(image, self.shape)
            h1, w1 = get_image_hw(image)
            
            # NOTE: Assign image info if it has not been defined
            # (just to be sure)
            info        = ImageInfo.from_file(image_path=path, info=info)
            info.height = h1 if info.height != h1 else info.height
            info.width  = w1 if info.width != w1 else info.width
            info.depth  = (image.shape[2] if info.depth != image.shape[2]
                           else info.depth)

        return image, info
    
    def load_mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load 4 images and target images and create a mosaic.
        
        Args:
            index (int):
                Index.
                
        Returns:
            input4 (np.ndarray):
                Mosaic input.
            target4 (np.ndarray):
                Mosaic target.
        """
        shape         = self.shape
        yc, xc        = shape[0], shape[1]  # Mosaic center x, y
        mosaic_border = [-yc // 2, -xc // 2]
        # 3 additional input indices
        indices = [index] + [int(torch.randint(len(self.data) - 1, (1,))) for _ in range(3)]
        
        # NOTE: Create mosaic input and target input
        for i, index in enumerate(indices):
            # Load input
            input, info = self.load_image(index=index)
            target, _   = self.load_enhance_image(index=index)
            h, w, _     = info.shape
            
            # Place input in input4
            if i == 0:  # Top left
                input4  = np.full((yc * 2, xc * 2, input.shape[2]), 114, np.uint8)
                # base input with 4 tiles
                target4 = np.full((yc * 2, xc * 2, target.shape[2]), 114, np.uint8)
                # base input with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # xmin, ymin, xmax, ymax (large input)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                # xmin, ymin, xmax, ymax (small input)
            elif i == 1:  # Top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, xc * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # Bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, xc * 2), min(yc * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            input4[y1a:y2a, x1a:x2a]  = input[y1b:y2b, x1b:x2b]
            # input4[ymin:ymax, xmin:xmax]
            target4[y1a:y2a, x1a:x2a] = target[y1b:y2b, x1b:x2b]
            # input4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
        
        return input4, target4
    
    def load_mixup(
        self, input: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MixUp https://arxiv.org/pdf/1710.09412.pdf."""
        input2, target2 = self.load_mosaic(
            index=int(torch.randint(len(self.data) - 1, (1,)))
        )
        ratio  = np.random.beta(8.0, 8.0)
        # mixup ratio, alpha=beta=8.0
        input  = input * ratio + input2 * (1 - ratio)
        input  = input.astype(np.uint8)
        target = target * ratio + target2 * (1 - ratio)
        target = target.astype(np.uint8)
        return input, target
        
    # MARK: Post-Load Data
    
    def post_load_data(self):
        """Post load data operations. We prepare `batch_shapes` for
        `rect_training` augmentation, and some labels statistics. If you want
        to add more operations, just `extend` this method.
        """
        # NOTE: Write data to our custom label format
        if not self.has_custom_labels and self.write_labels:
            self.write_custom_labels()
    
    # MARK: Get Item
    
    def get_item(self, index: int) -> tuple[Tensor, Tensor, Int3T]:
        """Get the item.
  
        Args:
            index (int):
                Index.
  
        Returns:
            input (Tensor[1, C, H, W]):
                Image.
            target (Tensor[1, C, H, W]):
                Enhanced image.
            shape (Int3T):
                Shape of the resized images.
        """
        input = target = shape = None
        
        # NOTE: Augmented load input
        if isinstance(self.load_augment, dict):
            mosaic = self.load_augment.get("mosaic", 0)
            mixup  = self.load_augment.get("mixup",  0)
            if torch.rand(1) <= mosaic:  # Load mosaic
                input, target = self.load_mosaic(index)
                shape         = input.shape
                if torch.rand(1) <= mixup:  # Mixup
                    input, target = self.load_mixup(input, target)
        
        # NOTE: Load input normally
        if input is None:
            input , info = self.load_image(index=index)
            target, _    = self.load_enhance_image(index=index)
            (h0, w0, _)  = info.shape0
            (h,  w,  _)  = info.shape
            shape        = (h0, w0), (h, w)
        
        # NOTE: Convert to tensor
        input  = to_tensor(input,  keep_dims=False, normalize=False).to(torch.uint8)
        target = to_tensor(target, keep_dims=False, normalize=False).to(torch.uint8)
        
        return input, target, shape
    
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, Int3T]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the DataLoader wrapper.
        """
        input, target, shapes = zip(*batch)  # Transposed

        if all(i.ndim == 3 for i in input) and all(t.ndim == 3 for t in target):
            input  = torch.stack(input,  0)
            target = torch.stack(target, 0)
        elif all(i.ndim == 4 for i in input) and all(t.ndim == 4 for t in target):
            input  = torch.cat(input,  0)
            target = torch.cat(target, 0)
        else:
            raise ValueError(f"Each `input.ndim` and `target.ndim` must be 3 or 4.")
            
        return input, target, shapes
        
    # MARK: Utils
    
    def write_custom_labels(self):
        """Write all data to custom label files using our custom label format.
        """
        # NOTE: Get label files
        dirnames = [os.path.dirname(p) for p in self.custom_label_paths]
        create_dirs(paths=dirnames)
        
        # NOTE: Scan all images and target images to get information
        with progress_bar() as pbar:
            for i in pbar.track(
                range(len(self.data)), description="[red]Scanning images"
            ):
                # image, hw_original, hw_resized
                _, self.data[i].image_info = self.load_image(index=i)
            
            for i in pbar.track(
                range(len(self.data)), description="[red]Scanning target images"
            ):
                # image, hw_original, hw_resized
                _, self.data[i].eimage_info = self.load_enhance_image(index=i)
            
            # NOTE: Parallel write labels
            for (data, path) in pbar.track(
                zip(self.data, self.custom_label_paths),
                description="[red]Writing custom annotations",
                total=len(self.data)
            ):
                VisionDataHandler().dump_to_file(data=data, path=path)
