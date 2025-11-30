#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Snow100K dataset.

This module implements Snow100K dataset for image desnowing tasks.
"""

__all__ = [
    "Snow100K",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="snow100k")
class Snow100K(ImageDataset):
    """Snow100K dataset."""

    _root_name : str         = "snow100k"
    _tasks     : list[Task]  = [Task.DESNOW]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classes   : Classes     = None
    
    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / self.split_str / "lq"]
        
        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))
        
        return images
