#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the SICE dataset.

This module implements the SICE dataset for exposure enhancement tasks.

References:
    - Paper: "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure
      Images," TIP 2018.
    - Code: https://github.com/csjcai/SICE

Notices:
    The testing index in Dataset_part1:
        - 4-23
        - 28
        - 31
        - 33-34
        - 37-39
        - 46-52
        - 55-69
        - 75-79
        - 100-103
    For the under-exposure testing, we choose the -1ev as the low-light input image:
        - If there are 7 images, then it is number 3.
        - If there are 9 images, then it is number 4.
    For the over-exposure testing, we choose the +1ev as the over-exposure input image:
        - If there are 7 images, then it is number 5.
        - If there are 9 images, then it is number 6. (My assumption)
"""

__all__ = [
    "SICE",
    "SICEME",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="sice")
class SICE(ImageDataset):
    """SICE dataset. We use the under-exposure images as the primary input
    modality.
    """
    
    _subset    : str         = "sice"
    _tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF, Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image"      : Modality(name="image_under", type="image", module=Image,           train=True, test=True, primary=True),
        "image_under": Modality(name="image_under", type="image", module=Image,           train=True, test=False),
        "image_over" : Modality(name="image_over",  type="image", module=Image,           train=True, test=False),
        "depth"      : Modality(name=DepthName,     type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"        : Modality(name="ref",         type="image", module=Image,           train=True, test=True),
    }
    _classes   : Classes     = None
    
    def __init__(self, lr: bool = True, *args, **kwargs):
        """Initializes the SICE dataset.
        
        Args:
            lr (bool): If True, use the low-resolution version of the dataset.
                Default is True.
        """
        self.lr = lr
        super().__init__(*args, **kwargs)
    
    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        if self.lr:
            patterns = [self.root / "sice_lr" / self.split_str / "image_under"]
        else:
            patterns = [self.root / "sice"    / self.split_str / "image_under"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))
        
        return images


@DATASETS.register(name="siceme")
class SICEME(ImageDataset):
    """SICE-ME dataset includes multi-exposure training images. This dataset is
    used in unsupervised curve-estimation methods for low-light enhancement
    (e.g., Zero-DCE, Zero-DCE++, etc.).
    """
    
    _subset    : str         = "sice"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True,  test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True,  test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=False, test=True),
    }
    _classes   : Classes     = None
    
    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / "me" / self.split_str / "image"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))
    
        return images
