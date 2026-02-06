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

from typing import override

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="realblur")
class RealBlurJ(ImageDataset, RegistrableMixin):
    """RealBlur-J dataset."""

    name: str = "realblur"
    tasks: list[Task] = [Task.DEBLUR]
    subroot: str = None
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities = {
        "image": Modality(
            name="image",
            type="image",
            module=Image,
            train=True,
            test=True,
            primary=True,
        ),
        "ref": Modality(
            name="ref",
            type="image",
            module=Image,
            train=True,
            test=False,
        ),
    }
    classlist: ClassList = None

    # --- Data Loading ---
    @override
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.

        Returns:
            A list of Image instances for the primary modality.
        """
        pattern = self.root / self.split_str / "j" / "image"

        images = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register(name="realblurr")
class RealBlurR(ImageDataset, RegistrableMixin):
    """RealBlur-R dataset."""

    name: str = "realblur"
    tasks: list[Task] = [Task.DEBLUR]
    subroot: str = None
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities = {
        "image": Modality(
            name="image",
            type="image",
            module=Image,
            train=True,
            test=True,
            primary=True,
        ),
        "ref": Modality(
            name="ref",
            type="image",
            module=Image,
            train=True,
            test=False,
        ),
    }
    classlist: ClassList = None

    # --- Data Loading ---
    @override
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.

        Returns:
            list[Any]: List of primary modality data files.
        """
        pattern = self.root / self.split_str / "r" / "image"

        images = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))

        return images


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
