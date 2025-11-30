#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for RealBlur dataset.

This module provides classes for the RealBlur-J and RealBlur-R datasets,
which are used for image deblurring tasks.
"""

__all__ = [
    "RealBlurJ",
    "RealBlurR",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="realblurj")
class RealBlurJ(ImageDataset):
    """RealBlur-J dataset."""
    
    _root_name : str         = "realblur"
    _tasks     : list[Task]  = [Task.DEBLUR]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classes   : Classes     = None

    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / self.split_str / "j" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register(name="realblurr")
class RealBlurR(ImageDataset):
    """RealBlur-R dataset."""

    _root_name : str         = "realblur"
    _tasks     : list[Task]  = [Task.DEBLUR]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classes   : Classes     = None

    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / self.split_str / "r" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images
