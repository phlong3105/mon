#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Snow100K dataset.

This module provides Snow100K dataset for image de-snowing.
"""

from __future__ import annotations

__all__ = [
    "Snow100K",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Snow100K(ImageDataset, RegistrableMixin):
    """Snow100K dataset."""

    name      : str         = "snow100k"
    tasks     : list[Task]  = [Task.DESNOW]
    subset    : str         = None
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
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
    classlist : ClassList   = None

    # --- Data Loading ---
    def _load_primary_data(self) -> list[Image]:
        """Load primary modality data files in the dataset.

        Returns:
            A list of Image instances for the primary modality.
        """
        pattern = self.root / self.split_str / "lq"

        images  = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
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
