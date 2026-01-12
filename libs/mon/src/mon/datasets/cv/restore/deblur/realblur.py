#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RealBlur dataset.

This module provides the RealBlur dataset for image de-blurring.
"""

from __future__ import annotations

__all__ = [
    "RealBlurJ",
    "RealBlurR",
]


from ....api import *


@DATASETS.register()
class RealBlurJ(ImageDataset, RegistrableMixin):
    """RealBlur-J dataset."""
    
    _name      : str         = "realblur"
    _tasks     : list[Task]  = [Task.DEBLUR]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
    }
    _classlist : ClassList   = None

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        pattern = self.root / self.split_str / "j" / "image"

        images  = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register()
class RealBlurR(ImageDataset, RegistrableMixin):
    """RealBlur-R dataset."""

    _name      : str         = "realblur"
    _tasks     : list[Task]  = [Task.DEBLUR]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
    }
    _classlist : ClassList   = None

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.
        
        Returns:
            A list of Image instances for the primary modality.
        """
        pattern = self.root / self.split_str / "r" / "image"

        images  = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))

        return images
