#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for UHD datasets.

This module implements UHD datasets for low-light enhancement tasks.
"""

__all__ = [
    "UHD4K",
    "UHD8K",
]

from mon.core import rich
from ....core import *


# @DATASETS.register(name="uhd4k")
class UHD4K(ImageDataset):
    """UHD-4K dataset."""
    
    _subset    : str         = "uhd"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : ClassList   = None

    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / "4k" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images


# @DATASETS.register(name="uhd8k")
class UHD8K(ImageDataset):
    """UHD-8K dataset."""
    
    _subset    : str         = "uhd"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : ClassList   = None

    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / "8k" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        return images
