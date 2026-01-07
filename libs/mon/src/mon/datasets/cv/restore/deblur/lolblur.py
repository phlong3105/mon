#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-Blur dataset.

This module implements LOL-Blur dataset for image deblurring, denoising, and
low-light enhancement.
"""

__all__ = [
    "LOLBlurB",
    "LOLBlurBN",
    "LOLBlurL",
    "LOLBlurLB",
    "LOLBlurLBN",
    "LOLBlurN",
]

import abc

from mon.core import rich
from ....api import *


class LOLBlur(ImageDataset, abc.ABC):
    """LOL-Blur dataset."""
    
    _subset    : str         = "lolblur"
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",      type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName,    type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",        type="image", module=Image,           train=True, test=True),
    }
    _classlist : ClassList   = None


@DATASETS.register(name="lolblurb")
class LOLBlurB(LOLBlur):
    """LOL-Blur-B (Blur) dataset."""

    _tasks: list[Task] = [Task.DEBLUR]
    
    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        patterns = [self.root / "b" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register(name="lolblurbn")
class LOLBlurBN(LOLBlur):
    """LOL-Blur-BN (Blur + Noise) dataset."""

    _tasks: list[Task] = [Task.DEBLUR, Task.DENOISE]

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        patterns = [self.root / "bn" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register(name="lolblurl")
class LOLBlurL(LOLBlur):
    """LOL-Blur-L (Low-Light) dataset."""

    _tasks: list[Task] = [Task.LLE]

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        patterns = [self.root / "l" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))
        
        return images


@DATASETS.register(name="lolblurlb")
class LOLBlurLB(LOLBlur):
    """LOL-Blur-LB (Low-Light + Blur) dataset."""

    _tasks: list[Task] = [Task.DEBLUR, Task.LLE]

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        patterns = [self.root / "lb" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))
        
        return images


@DATASETS.register(name="lolblurlbn")
class LOLBlurLBN(LOLBlur):
    """LOL-Blur-LBN (Low-Light + Blur + Noise) dataset."""

    _tasks: list[Task] = [Task.DEBLUR, Task.DENOISE, Task.LLE]

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        patterns = [self.root / "lbn" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register(name="lolblurn")
class LOLBlurN(LOLBlur):
    """LOL-Blur-N (Noise) dataset."""

    _tasks: list[Task] = [Task.DENOISE]

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        patterns = [self.root / "n" / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images
