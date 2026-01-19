#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NTIRE 2025 LLIE dataset.

This module provides the NTIRE 2025 LLIE dataset for low-light image enhancement.

References:
    - Data: https://codalab.lisn.upsaclay.fr/competitions/21636
"""

from __future__ import annotations

__all__ = [
    "NTIRE2025LLIE",
]


from ...api import *


@DATASETS.register()
class NTIRE2025LLIE(ImageDataset, RegistrableMixin):
    """NTIRE 2025 LLIE dataset."""

    _name      : str         = "ntire2025llie"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
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

        Raises:
            ValueError: If the specified ``split`` is invalid.
        """
        if self.split in [Split.TRAIN]:
            patterns = [self.root / "train" / "image"]
        elif self.split in [Split.VAL]:
            patterns = [self.root / "val" / "image"]
        elif self.split in [Split.TEST]:
            patterns = [self.root / "test" / "image"]
        else:
            raise ValueError(f"``split`` invalid: [{self.split}]")

        images = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
