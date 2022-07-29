#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Template for (Supervised) Image Classification datasets.
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
    "ImageClassificationDataset",
]


# MARK: - ImageClassificationDataset

class ImageClassificationDataset(VisionDataset, metaclass=ABCMeta):
    """A base class for all (supervised) image classification datasets.
    
    Attributes:
        root (str):
            Dataset root directory that contains: train/val/test/...
            subdirectories.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        image_paths (list):
            List of all image files.
        label_paths (list):
            List of all label files.
        custom_label_paths (list):
            List of all custom label files.
        data (list[VisionData]):
            List of all `VisionData` objects.
        class_labels (ClassLabels, optional):
            `ClassLabels` object contains all class-labels defined in the
            dataset.
        shape (Int3T):
            Image shape as [H, W, C]
        batch_size (int):
            Number of training samples in one forward & backward pass.
        batch_shapes (np.ndarray, optional):
            Array of batch shapes. It is available only for `rect_training`
            augmentation.
        batch_indexes (np.ndarray, optional):
            Array of batch indexes. It is available only for `rect_training`
            augmentation.
        caching_labels (bool):
            Should overwrite the existing cached labels?
        caching_images (bool):
            Cache images into memory for faster training.
        write_labels (bool):
            After loading images and labels for the first time, we will convert
            it to our custom data format and write to files. If `True`, we will
            overwrite these files.
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
            
    Work Flow:
        __init__()
            |__ list_files()
            |__ load_data()
            |		|__ cache_labels()
            |		|		|__ load_label()
            |		|		|__ load_labels()
            |		|__ cache_images()
            |				|__ load_images()
            |__ post_load_data()
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str,
        class_labels    : Optional[ClassLabels]   = None,
        shape           : Int3T                   = (640, 640, 3),
        batch_size      : int                     = 1,
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
            root			 = root,
            transforms       = transforms,
            transform        = transform,
            target_transform = target_transform
        )
        self.split              = split
        self.class_labels       = class_labels
        self.shape              = shape
        self.batch_size         = batch_size
        self.batch_shapes       = None
        self.batch_indexes      = None
        self.caching_labels     = caching_labels
        self.caching_images     = caching_images
        self.write_labels       = write_labels
        self.fast_dev_run       = fast_dev_run
        self.load_augment       = load_augment
        self.augment 		    = augment

        if vision_backend in VisionBackend:
            self.vision_backend = vision_backend
        else:
            self.vision_backend = VISION_BACKEND

        self.image_paths        = []
        self.label_paths        = []
        self.custom_label_paths = []
        self.data               = []

        # NOTE: List files
        self.list_files()
        # NOTE: Load classlabels
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
            input  = self.augment.forward(input=input)
        else:
            input  = to_tensor(input, normalize=True)
            
        if self.transform is not None:
            input  = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            input  = self.transforms(input)
            target = self.transforms(target)
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
    def image_size(self) -> int:
        """Return image size."""
        return max(self.shape)
    
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
              `image_paths`, `label_paths`, `custom_label_paths`.
        """
        pass

    # MARK: Load ClassLabels
    
    @abstractmethod
    def load_class_labels(self):
        """Load ClassLabels."""
        pass
    
    # MARK: Load Data
    
    def load_data(self):
        """Load labels, cache labels and images."""
        # NOTE: Cache labels
        """
        file = (
            self.label_paths if isinstance(self.label_paths, str)
            else self.label_paths[0]
        )
        split_prefix      = file[: file.find(self.split)]
        cached_label_path = f"{split_prefix}{self.split}.cache"
        """
        cached_label_path = os.path.join(self.root, f"{self.split}.cache")

        if os.path.isfile(cached_label_path):
            cache = torch.load(cached_label_path)  # Load
            hash  = (get_hash([self.label_paths] + self.custom_label_paths + self.image_paths)
                     if isinstance(self.label_paths, str)
                     else get_hash(self.label_paths + self.custom_label_paths + self.image_paths))
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
        # NOTE: Load all labels from a `.json` file
        if isinstance(self.label_paths, str) and os.path.isfile(self.label_paths):
            cache_labels = self.load_labels(self.image_paths, self.label_paths)
        # NOTE: Load each pair of image file and label file together
        else:
            cache_labels = {}

            with progress_bar() as pbar:
                for i in pbar.track(
                    range(len(self.image_paths)),
                    description=f"[bright_yellow]Caching {self.split} labels"
                ):
                    label_path 	      = self.label_paths[i]
                    custom_label_path = (
                        self.custom_label_paths[i]
                        if (
                            self.has_custom_labels
                            and os.path.isfile(self.custom_label_paths[i])
                        )
                        else None
                    )
                    label = self.load_label(
                        image_path 	 	  = self.image_paths[i],
                        label_path	 	  = label_path,
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
        if isinstance(self.label_paths, str):
            cache_labels["hash"] = get_hash(
                [self.label_paths] + self.custom_label_paths + self.image_paths
            )
        else:
            cache_labels["hash"] = get_hash(
                self.label_paths + self.custom_label_paths + self.image_paths
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
        # NOTE: Load all labels from a `.json` file
        if isinstance(self.label_paths, str) and os.path.isfile(self.label_paths):
            cache_labels = self.load_labels(self.image_paths, self.label_paths)
        # NOTE: Load each pair of image file and label file together
        else:
            cache_labels = {}
            total        = len(self.image_paths)
        
            with progress_bar() as pbar:
                task = pbar.add_task(
                    f"[bright_yellow]Caching {self.split} labels", total=total
                )
                
                def cache_label(i):
                    label_path 	      = self.label_paths[i]
                    custom_label_path = (
                        self.custom_label_paths[i]
                        if (
                            self.has_custom_labels
                            and os.path.isfile(self.custom_label_paths[i])
                        )
                        else None
                    )
                    label = self.load_label(
                        image_path 	 	  = self.image_paths[i],
                        label_path	 	  = label_path,
                        custom_label_path = custom_label_path
                    )
                    cache_labels[self.image_paths[i]] = label
                    pbar.update(task, advance=1)
                
                Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
                    delayed(cache_label)(i) for i in range(total)
                )
        
        # NOTE: Check for any changes btw the cached labels
        if isinstance(cache_labels, dict):
            cache_labels = SortedDict(cache_labels)
        for (k, v) in cache_labels.items():
            if v.image_info.path != k:
                self.caching_labels = True
                break
        
        # NOTE: Write cache
        console.log(f"Labels has been cached to: {path}.")
        if isinstance(self.label_paths, str):
            cache_labels["hash"] = get_hash(
                [self.label_paths] + self.custom_label_paths + self.image_paths
            )
        else:
            cache_labels["hash"] = get_hash(
                self.label_paths + self.custom_label_paths + self.image_paths
            )
        torch.save(cache_labels, path)  # Save for next time
        return cache_labels
    
    @abstractmethod
    def load_label(
        self,
        image_path		 : str,
        label_path		 : str,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load label data associated with the image from the corresponding
        label file.

        Args:
            image_path (str):
                Image file.
            label_path (str):
                Label file.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.

        Returns:
            data (VisionData):
                `VisionData` object.
        """
        pass
    
    @abstractmethod
    def load_labels(
        self,
        image_paths	     : list[str],
        label_path 	     : str,
        custom_label_path: Optional[str] = None
    ) -> dict[str, VisionData]:
        """Load all labels from one label file.

        Args:
            image_paths (list[str]):
                List of image paths.
            label_path (str):
                Label file.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.

        Returns:
            data (dict):
                Dictionary of `VisionData` objects.
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
                range(len(self.image_paths)), description="[red]Caching images"
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
            
            # NOTE: Assign image info if it has not been defined (just to be sure)
            info        = ImageInfo.from_file(image_path=path, info=info)
            info.height = h1 if info.height != h1 else info.height
            info.width  = w1 if info.width  != w1 else info.width
            info.depth  = (image.shape[2] if info.depth != image.shape[2]
                           else info.depth)
            
        return image, info
    
    # MARK: Post-Load Data
    
    def post_load_data(self):
        """Post load data operations. We prepare `batch_shapes` for
        `rect_training` augmentation, and some labels statistics. If you want
        to add more operations, just `extend` this method.
        """
        # NOTE: Prepare for rectangular training
        # if self.augment.rect:
        #    self.prepare_for_rect_training()
        
        # NOTE: Write data to our custom label format
        if not self.has_custom_labels and self.write_labels:
            self.write_custom_labels()
    
    # MARK: Get Item
    
    def get_item(self, index: int) -> tuple[Tensor, Tensor, Int3T]:
        """Convert the data at the given index to enhancement input item.
  
        Args:
            index (int):
                Index.
  
        Returns:
            input (Tensor[1, C, H, W]):
                Image.
            target (Tensor[1]):
                Class ID.
            shape (Int3T):
                Shape of the resized images.
        """
        data = self.data[index]
        
        # NOTE: Load image normally
        input, info = self.load_image(index=index)
        (h0, w0, _) = info.shape0
        (h,  w,  _) = info.shape
        shape       = (h0, w0), (h, w)
        target      = data.class_id
        target      = target if isinstance(target, int) else int(target)
        
        # NOTE: Convert to tensor
        input  = to_tensor(input, keep_dims=False, normalize=False).to(torch.uint8)
        target = torch.tensor(target)
        
        return input, target, shape
        
    @staticmethod
    def collate_fn(batch) -> tuple[Tensor, Tensor, Int3T]:
        """Collate function used to fused input items together when using
        `batch_size > 1`. This is used in the DataLoader wrapper.
        """
        input, target, shapes = zip(*batch)  # transposed
        if all(i.ndim == 3 for i in input):
            input = torch.stack(input, 0)
        elif all(i.ndim == 4 for i in input):
            input = torch.cat(input, 0)
        else:
            raise ValueError(f"Each `input.ndim` must be 3 or 4.")
        
        return input, torch.cat(target), shapes
        
    # MARK: Utils
    
    def prepare_for_rect_training(self):
        """Prepare `batch_shapes` for `rect_training` augmentation.
        
        References:
            https://github.com/ultralytics/yolov3/issues/232
        """
        rect   = self.augment.rect
        stride = self.augment.stride
        pad    = self.augment.pad
        
        if rect:
            # NOTE: Get number of batches
            n  = len(self.data)
            bi = np.floor(np.arange(n) / self.batch_size).astype(np.int)  # Batch index
            nb = bi[-1] + 1  # Number of batches
            
            # NOTE: Sort data by aspect ratio
            s     = [data.image_info.shape0 for data in self.data]
            s     = np.array(s, dtype=np.float64)
            ar    = s[:, 1] / s[:, 0]  # Aspect ratio
            irect = ar.argsort()
            
            self.image_paths = [self.image_paths[i] for i in irect]
            self.label_paths = [self.label_paths[i] for i in irect]
            self.data        = [self.data[i]        for i in irect]
            ar				 = ar[irect]
            
            # NOTE: Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            
            self.batch_shapes  = stride * np.ceil(
                np.array(shapes) * self.image_size / stride + pad).astype(np.int)
            self.batch_indexes = bi
        
    def write_custom_labels(self):
        """Write all data to custom label files using our custom label format.
        """
        # NOTE: Get label files
        """
        parents     = [str(Path(file).parent) for file in self.label_paths]
        stems       = [str(Path(file).stem)   for file in self.label_paths]
        stems       = [stem.replace("_custom", "") for stem in stems]
        label_paths = [os.path.join(parent, f"{stem}_custom.json")
                       for (parent, stem) in zip(parents, stems)]
        """
        dirnames = [os.path.dirname(p) for p in self.custom_label_paths]
        create_dirs(paths=dirnames)
        
        # NOTE: Scan all images to get information
        with progress_bar() as pbar:
            for i in pbar.track(
                range(len(self.data)), description="[red]Scanning images"
            ):
                # image, hw_original, hw_resized
                _, self.data[i].image_info = self.load_image(index=i)
            
            # NOTE: Parallel write labels
            for (data, path) in pbar.track(
                zip(self.data, self.custom_label_paths),
                description="[red]Writing custom annotations",
                total=len(self.data)
            ):
                VisionDataHandler().dump_to_file(data=data, path=path)
