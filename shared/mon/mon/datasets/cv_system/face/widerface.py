#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for WiderFace dataset.

This module implements the WiderFace dataset for face detection tasks.
"""

__all__ = [
    "WiderFace",
    "WiderFaceVal",
]

from mon.core import rich
from ...core import *


@DATASETS.register(name="widerface")
class WiderFace(ImageDataset):
    """WiderFace dataset."""
    
    _root_name : str         = "widerface"
    _tasks     : list[Task]  = [Task.DETECT]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classes   : Classes     = Classes([
        {"name": "face", "id": 0, "color": [ 81, 120, 228]},
    ])


@DATASETS.register(name="widerfaceval")
class WiderFaceVal(WiderFace):
    """WiderFace-Val subset."""
    
    def _load_primary_data(self) -> list[Image]:
        """Lists all image data for the primary modality.
        
        Returns:
            list[Image]: A list of Image instances for the primary modality.
        """
        patterns = [self.root / "val" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(data=path, root=pattern))

        return images
