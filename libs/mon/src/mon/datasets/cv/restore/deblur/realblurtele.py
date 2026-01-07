#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RealBlurTele dataset.

This module implements the RealBlurTele dataset for image de-blurring.
"""

__all__ = [
    "RealBlurTeleJ",
    "RealBlurTeleR",
]

from mon.core import rich
from ....api import *


@DATASETS.register(name="realblurtelej")
class RealBlurTeleJ(ImageDataset):
    """RealBlurTele-J dataset."""
    
    _subset    : str         = "realblurtele"
    _tasks     : list[Task]  = [Task.DEBLUR]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classlist : ClassList   = None

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
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


@DATASETS.register(name="realblurteler")
class RealBlurTeleR(ImageDataset):
    """RealBlurTele-R dataset."""

    _subset    : str         = "realblurtele"
    _tasks     : list[Task]  = [Task.DEBLUR]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classlist : ClassList   = None

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
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
