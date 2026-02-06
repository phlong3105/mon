#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain1200 dataset.

This module provides the Rain1200 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain1200",
]

from typing import override

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="rain1200")
class Rain1200(ImageDataset, RegistrableMixin):
    """Rain1200 dataset."""

    name: str = "rain1200"
    tasks: list[Task] = [Task.DERAIN]
    subroot: str = None
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
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
            test=True,
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
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / self.split_str / "light" / "image",
                self.root / self.split_str / "medium" / "image",
                self.root / self.split_str / "heavy" / "image",
            ]
        else:
            patterns = [self.root / self.split_str / "image"]

        images = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
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
