#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Rain1200 dataset.

This module implements the Rain1200 dataset for image deraining tasks.
"""

__all__ = [
    "Rain1200",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="rain1200")
class Rain1200(ImageDataset):
    """Rain1200 dataset."""

    _root_name : str         = "rain1200"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : Classes     = None
    
    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / self.split_str / "light"  / "image",
                self.root / self.split_str / "medium" / "image",
                self.root / self.split_str / "heavy"  / "image",
            ]
        else:
            patterns = [
                self.root / self.split_str / "image",
            ]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))
      
        return images
