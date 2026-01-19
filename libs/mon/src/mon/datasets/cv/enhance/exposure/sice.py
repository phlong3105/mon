#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SICE dataset.

This module provides the SICE dataset for exposure enhancement.

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

from __future__ import annotations

__all__ = [
    "SICE",
    "SICEME",
]


from ....api import *


@DATASETS.register()
class SICE(ImageDataset, RegistrableMixin):
    """SICE dataset.

    We use the under-exposure images as the primary input modality.
    """

    _name      : str         = "sice"
    _tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF, Task.LLE]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image"      : Modality(
            name    = "image_under",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "image_under": Modality(
            name    = "image_under",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
        "image_over" : Modality(
            name    = "image_over",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
        "depth"      : Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"        : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None

    # --- Lifecycle & Initialization ---
    def __init__(self, lr: bool = True, *args, **kwargs):
        """Initializes a new instance.

        Args:
            lr: If True, use the low-resolution version of the dataset.
                Default is True.
        """
        self.lr = lr
        super().__init__(*args, **kwargs)

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.

        Returns:
            A list of Image instances for the primary modality.
        """
        base_dir = "sice_lr" if self.lr else "sice"
        pattern  = self.root / base_dir / self.split_str / "image_under"

        images   = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register()
class SICEME(ImageDataset, RegistrableMixin):
    """SICE-ME dataset includes multi-exposure training images.

    This dataset is used in unsupervised curve-estimation methods for low-light
    enhancement (e.g., Zero-DCE, Zero-DCE++, etc.).
    """

    _name      : str         = "siceme"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "me"
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
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = False,
            test    = True,
        ),
    }
    _classlist : ClassList   = None


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
