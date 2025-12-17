#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for MIPI 2024 Flare dataset.

This module implements the MIPI 2024 Flare dataset for image deflare tasks.

References:
	- Data: https://mipi-challenge.org/MIPI2024/index.html
"""

__all__ = [
	"MIPI2024Flare",
]

from mon.core import rich
from ...core import *


@DATASETS.register(name="mipi2024flare")
class MIPI2024Flare(ImageDataset):
    """MIPI 2024 Flare dataset."""
    
    _subset    : str         = "mipi2024flare"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classes   : Classes     = None
    
    # ----- Initialize -----
    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
            
        Raises:
            ValueError: If the specified split is invalid.
        """
        if self.split in [Split.TRAIN]:
            patterns = [self.root / "train" / "image"]
        elif self.split in [Split.VAL]:
            patterns = [self.root / "val"   / "image"]
        elif self.split in [Split.TEST]:
            patterns = [self.root / "test"  / "image"]
        else:
            raise ValueError(f"``split`` invalid: [{self.split}]")
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images
