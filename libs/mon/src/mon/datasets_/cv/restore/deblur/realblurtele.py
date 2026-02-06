#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RealBlurTele dataset.

This module provides the RealBlurTele dataset for image de-blurring.
"""

from __future__ import annotations

__all__ = [
    "RealBlurTeleJ",
    "RealBlurTeleR",
]

from typing_extensions import override

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="realblurtelej")
class RealBlurTeleJ(ImageDataset, RegistrableMixin):
    """RealBlurTele-J dataset."""

    name: str = "realblurtelej"
    tasks: list[Task] = [Task.DEBLUR]
    subroot: str = "j"
    splits: list[Split] = [Split.TEST]
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
        pattern = self.root / self.split_str / "j" / "image"

        images = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    images.append(Image(data=path, root=pattern))

        return images


@DATASETS.register(name="realblurteler")
class RealBlurTeleR(ImageDataset, RegistrableMixin):
    """RealBlurTele-R dataset."""

    name: str = "realblurteler"
    tasks: list[Task] = [Task.DEBLUR]
    subroot: str = "r"
    splits: list[Split] = [Split.TEST]
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
